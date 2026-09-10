"""
Submit a whole-slide expression run to a Slurm cluster, and fetch the result.

The transcriptome-wide pass is a GPU job, so this drives it where the GPU is:
copy what is needed, submit to the queue, wait, bring back the map. The browser
then loads that map next to the slide as if it had computed it.

Authentication is entirely your SSH config, keys and agent. This never asks for
a password and never stores one — if `ssh cluster` works in your terminal, this
works, and if it does not, fix it there first.

Used through predict_expression.py:

    python scripts/predict_expression.py slide.svs --all \\
        --submit cluster --partition gpuq

Add --dry-run to see the sbatch script and every command, and send nothing.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

JOB_ID = re.compile(r"Submitted batch job (\d+)")

# Slurm states that mean the job is over, whether or not it worked.
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}


@dataclass
class Job:
    host: str
    remote_dir: str
    partition: str = "gpuq"
    gres: str = "gpu:1"
    time_limit: str = "08:00:00"
    cpus: int = 8
    mem: str = "64G"
    python: str = "python"
    account: str | None = None
    name: str = "slidecraft-expr"
    modules: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def sbatch_script(job: Job, slide_remote: str, out_remote: str, args: list[str]) -> str:
    """
    The batch script, written out in full rather than assembled on the node.

    Everything the run depends on is visible here — partition, resources, the
    interpreter, the exact command — because the usual failure on a cluster is
    that the job ran somewhere subtly different from where you thought, and a
    script you can read is how that gets caught.
    """
    directives = [
        f"#SBATCH --job-name={job.name}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --gres={job.gres}",
        f"#SBATCH --time={job.time_limit}",
        f"#SBATCH --cpus-per-task={job.cpus}",
        f"#SBATCH --mem={job.mem}",
        f"#SBATCH --output={job.remote_dir}/%j.out",
        f"#SBATCH --error={job.remote_dir}/%j.err",
    ]
    if job.account:
        directives.append(f"#SBATCH --account={job.account}")

    lines = ["#!/bin/bash", *directives, "", "set -euo pipefail", ""]
    lines += [f"module load {m}" for m in job.modules]
    # Exported rather than inlined into the command, so a token never lands in
    # the process table where `ps` on a shared login node would show it.
    lines += [f"export {k}={shlex.quote(v)}" for k, v in job.env.items()]
    lines += [
        "",
        f"cd {shlex.quote(job.remote_dir)}",
        'echo "node: $(hostname)"',
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
        "",
        " ".join(
            [
                shlex.quote(job.python),
                shlex.quote(f"{job.remote_dir}/predict_expression.py"),
                shlex.quote(slide_remote),
                *[shlex.quote(a) for a in args],
                "--out",
                shlex.quote(out_remote),
            ]
        ),
        "",
    ]
    return "\n".join(lines)


def parse_job_id(stdout: str) -> str | None:
    m = JOB_ID.search(stdout)
    return m.group(1) if m else None


def parse_state(stdout: str) -> str | None:
    """
    First state line from `sacct`/`squeue`.

    sacct reports one row per step — the job, then `.batch`, `.extern` and so on
    — and only the first is the job itself. Reading a later row is how a
    finished job gets mistaken for a running one.
    """
    for line in stdout.splitlines():
        state = line.strip().split("|")[0].strip().split()[0] if line.strip() else ""
        if state:
            return state.rstrip("+")
    return None


def expand_home(path: str, home: str) -> str:
    """
    Replace a leading `~` with the real remote home.

    Quoting a path for the remote shell is necessary — a directory with a space
    in it would break the command otherwise — but quoting also stops `~` being
    expanded, leaving a literal directory called `~` in the working directory.
    Slurm is worse: `#SBATCH --output=~/x.out` is never expanded at all, quoted
    or not, and the job fails to start with nothing useful in any log. So `~` is
    resolved here, once, against the home the cluster actually reports.
    """
    if path == "~":
        return home
    if path.startswith("~/"):
        return f"{home}/{path[2:]}"
    return path


class Runner:
    """Shells out, or prints what it would have done."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, argv: list[str], capture: bool = True, check: bool = True) -> str:
        printable = " ".join(shlex.quote(a) for a in argv)
        if self.dry_run:
            print(f"  $ {printable}")
            return ""
        print(f"$ {printable}", file=sys.stderr)
        # stdin and stderr are inherited so ssh can prompt for a key passphrase
        # and show host-key questions; only stdout is captured.
        proc = subprocess.run(argv, capture_output=capture, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"{printable}\nexit {proc.returncode}\n{(proc.stderr or '').strip()}"
            )
        return proc.stdout or ""


def submit(
    job: Job,
    slide: Path,
    args: list[str],
    *,
    remote_slide: str | None = None,
    watch: bool = True,
    poll: int = 30,
    dry_run: bool = False,
) -> int:
    r = Runner(dry_run)
    here = Path(__file__).resolve().parent

    print(f"Cluster: {job.host}  partition: {job.partition}  {job.gres}")
    r.run(["ssh", job.host, "true"], capture=False)

    if dry_run:
        home = "/home/$USER"
        print("  (dry run: assuming this remote home)")
    else:
        home = r.run(["ssh", job.host, 'printf %s "$HOME"']).strip()
        if not home:
            raise RuntimeError(f"Could not read $HOME on {job.host}")

    job.remote_dir = expand_home(job.remote_dir, home)
    job.python = expand_home(job.python, home)
    remote = PurePosixPath(job.remote_dir)

    r.run(["ssh", job.host, f"mkdir -p {shlex.quote(str(remote))}"])

    # The script travels with the job. A cluster copy that has drifted from the
    # one here produces results that do not match this checkout, and nothing
    # says so.
    r.run(["rsync", "-a", str(here / "predict_expression.py"), f"{job.host}:{remote}/"])

    if remote_slide:
        slide_path = expand_home(remote_slide, home)
        print(f"Using the slide already on the cluster: {slide_path}")
    else:
        # Slides are gigabytes; --partial and -z make a resumed copy cheap, and
        # rsync skips it entirely if it is already there and unchanged.
        slide_path = str(remote / slide.name)
        r.run(["rsync", "-az", "--partial", "--info=progress2", str(slide), f"{job.host}:{remote}/"])

    out_remote = str(remote / f"{slide.stem}.expression.bin")
    script = sbatch_script(job, slide_path, out_remote, args)

    if dry_run:
        print("\n--- sbatch script ---")
        print(script)
        print("--- end ---\n")

    script_remote = str(remote / "run.sbatch")
    # Piped in over stdin rather than written locally and copied, so there is no
    # temporary file to leave behind on either machine.
    if dry_run:
        print(f"  $ ssh {job.host} 'cat > {script_remote}' < <(the script above)")
    else:
        proc = subprocess.run(
            ["ssh", job.host, f"cat > {shlex.quote(script_remote)}"],
            input=script, text=True, capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Could not write the batch script: {proc.stderr.strip()}")

    out = r.run(["ssh", job.host, f"sbatch {shlex.quote(script_remote)}"])
    if dry_run:
        print("\nDry run: nothing was submitted.")
        return 0

    job_id = parse_job_id(out)
    if not job_id:
        raise RuntimeError(f"Could not read a job id from sbatch:\n{out.strip()}")
    print(f"Submitted job {job_id}. Logs: {remote}/{job_id}.out")

    if not watch:
        print(f"Not waiting. When it finishes:\n"
              f"  rsync -az {job.host}:{out_remote} .")
        return 0

    state = None
    while True:
        time.sleep(poll)
        raw = r.run(
            ["ssh", job.host,
             f"sacct -j {job_id} --format=State --noheader --parsable2 2>/dev/null "
             f"|| squeue -j {job_id} -h -o %T"],
            check=False,
        )
        next_state = parse_state(raw)
        if next_state and next_state != state:
            state = next_state
            print(f"  {state}")
        if state in TERMINAL:
            break
        if not next_state and state is not None:
            # Gone from both: sacct is not always configured, and a job that has
            # left the queue without a recorded state has finished somehow.
            print("  job left the queue")
            break

    if state and state != "COMPLETED":
        print(f"\nJob {job_id} ended as {state}. Tail of the error log:", file=sys.stderr)
        print(r.run(["ssh", job.host, f"tail -n 40 {remote}/{job_id}.err"], check=False), file=sys.stderr)
        return 1

    local_out = slide.parent / f"{slide.stem}.expression.bin"
    r.run(["rsync", "-az", "--info=progress2", f"{job.host}:{out_remote}", str(local_out)])
    print(f"\nWrote {local_out}")
    print("Drop its folder into Slidecraft; it attaches to the slide by name.")
    return 0
