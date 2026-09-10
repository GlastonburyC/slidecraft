"""
Checks for the cluster launcher's pure parts.

    .venv-export/bin/python -m pytest scripts/test_hpc.py -q

The SSH and Slurm calls are not exercised here — those need a cluster. What is
covered is everything that decides *what gets sent*, which is where a mistake is
silent: a mis-parsed state polls forever, a mis-quoted path fails on the node
with nothing useful in the log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hpc import Job, expand_home, parse_job_id, parse_state, sbatch_script  # noqa: E402


def test_expands_a_leading_tilde():
    assert expand_home("~/slidecraft", "/home/cg") == "/home/cg/slidecraft"
    assert expand_home("~", "/home/cg") == "/home/cg"


def test_leaves_absolute_and_relative_paths_alone():
    assert expand_home("/scratch/cg/run", "/home/cg") == "/scratch/cg/run"
    assert expand_home("work/run", "/home/cg") == "work/run"
    # A tilde that is not the home shorthand is a real directory name.
    assert expand_home("./~odd", "/home/cg") == "./~odd"


def test_reads_the_job_id_sbatch_prints():
    assert parse_job_id("Submitted batch job 1234567\n") == "1234567"
    assert parse_job_id("sbatch: error: invalid partition") is None


def test_reads_only_the_first_state_row():
    # sacct emits a row per step; the job's own state is the first, and reading
    # a later one mistakes a finished job for a running one.
    out = "COMPLETED\nCOMPLETED\nCOMPLETED\n"
    assert parse_state(out) == "COMPLETED"
    # Slurm marks cancellations with a trailing "+" and a user.
    assert parse_state("CANCELLED by 1234\n") == "CANCELLED"
    assert parse_state("RUNNING\n") == "RUNNING"
    assert parse_state("\n\n") is None


def _job(**kw):
    return Job(host="h", remote_dir="/home/cg/slidecraft", **kw)


def test_script_carries_the_requested_resources():
    s = sbatch_script(_job(partition="gpuq", gres="gpu:2", time_limit="12:00:00", mem="128G"),
                      "/data/a.svs", "/out/a.bin", ["--all"])
    assert "#SBATCH --partition=gpuq" in s
    assert "#SBATCH --gres=gpu:2" in s
    assert "#SBATCH --time=12:00:00" in s
    assert "#SBATCH --mem=128G" in s
    # Logs must be absolute: Slurm never expands ~ in an output path.
    assert "#SBATCH --output=/home/cg/slidecraft/%j.out" in s
    assert "~" not in s


def test_account_is_only_present_when_asked_for():
    assert "--account" not in sbatch_script(_job(), "/a.svs", "/o.bin", [])
    assert "#SBATCH --account=proj1" in sbatch_script(_job(account="proj1"), "/a.svs", "/o.bin", [])


def test_forwards_the_prediction_arguments_verbatim():
    s = sbatch_script(_job(), "/data/a.svs", "/out/a.bin", ["--all", "--batch", "64"])
    command = next(l for l in s.splitlines() if "predict_expression.py" in l)
    # shlex.quote leaves tokens that need no quoting alone, so compare the
    # parsed command rather than its spelling.
    import shlex as _shlex

    parts = _shlex.split(command)
    assert parts[1].endswith("predict_expression.py")
    assert parts[2] == "/data/a.svs"
    assert parts[3:] == ["--all", "--batch", "64", "--out", "/out/a.bin"]


def test_quotes_paths_with_spaces():
    import shlex as _shlex

    s = sbatch_script(_job(), "/data/TB 310.svs", "/out/TB 310.bin", [])
    command = next(l for l in s.splitlines() if "predict_expression.py" in l)
    # The point is that it survives the shell, whatever the quoting looks like.
    assert _shlex.split(command)[2] == "/data/TB 310.svs"


def test_secrets_are_exported_not_placed_on_the_command_line():
    s = sbatch_script(_job(env={"HF_TOKEN": "hf_secret"}), "/a.svs", "/o.bin", ["--all"])
    export_line = next(l for l in s.splitlines() if l.startswith("export HF_TOKEN="))
    command_line = next(l for l in s.splitlines() if "predict_expression.py" in l)
    # `ps` on a shared login node shows command lines; it does not show exports.
    assert "hf_secret" in export_line
    assert "hf_secret" not in command_line


def test_modules_load_before_the_run():
    s = sbatch_script(_job(modules=["cuda/12.4", "openslide"]), "/a.svs", "/o.bin", [])
    lines = s.splitlines()
    assert lines.index("module load cuda/12.4") < lines.index(
        next(l for l in lines if "predict_expression.py" in l)
    )


def test_the_script_stops_on_the_first_failure():
    # Without this a failed model load still exits 0 and the launcher fetches
    # a file that was never written, or an older one.
    assert "set -euo pipefail" in sbatch_script(_job(), "/a.svs", "/o.bin", [])


def test_ssh_options_reach_both_ssh_and_rsync():
    # rsync spawns its own ssh, which does not inherit ours — so a shared
    # connection that works for ssh but not rsync means the copies prompt for a
    # password while everything else sails through.
    from hpc import rsync_cmd, ssh_cmd

    job = _job(ssh_options=["-S", "/tmp/cm-host"])
    assert ssh_cmd(job, "true") == ["ssh", "-S", "/tmp/cm-host", "h", "true"]

    cmd = rsync_cmd(job, "-a", "x", "h:/y")
    assert cmd[:3] == ["rsync", "-e", "ssh -S /tmp/cm-host"]


def test_no_ssh_options_leaves_the_commands_plain():
    from hpc import rsync_cmd, ssh_cmd

    job = _job()
    assert ssh_cmd(job, "true") == ["ssh", "h", "true"]
    assert rsync_cmd(job, "-a", "x", "h:/y")[:3] == ["rsync", "-e", "ssh"]
