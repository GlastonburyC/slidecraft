/**
 * Render TUTORIAL.md into the site as a styled page.
 *
 * The tutorial is written once, in Markdown, because that is what people read
 * in the repository. The site needs the same words in the site's own skin, and
 * keeping two copies in step by hand is how they stop being in step — so the
 * page is generated, and `docs/TUTORIAL.html` is not edited by hand.
 *
 * A small converter rather than a Markdown dependency: this file's input is one
 * document we control, and the subset it uses is fixed.
 */
import { readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const escape = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Inline spans, applied after the block structure is decided. */
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, (_, c) => `<code>${escape(c)}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, t, h) => {
      const href = h.replace(/\.md$/, ".html");
      return `<a href="${href}">${t}</a>`;
    });
}

function render(md) {
  const out = [];
  const lines = md.split("\n");
  let i = 0;
  let inCode = false;
  let code = [];

  const flushCode = () => {
    out.push(`<pre><code>${escape(code.join("\n"))}</code></pre>`);
    code = [];
  };

  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith("```")) {
      if (inCode) { flushCode(); inCode = false; } else inCode = true;
      i++;
      continue;
    }
    if (inCode) { code.push(line); i++; continue; }

    if (/^\s*$/.test(line)) { i++; continue; }

    if (line.startsWith("---")) { out.push("<hr>"); i++; continue; }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1].length + 1; // h1 is the page title
      /*
       * Headings carry an id, so the site can link straight at one.
       *
       * Written explicitly as `## Title {#slug}` wherever something links to
       * it: a slug derived from the words changes the moment the words do, and
       * a tutorial link that silently stops landing anywhere is worse than no
       * link. Anything without one still gets a derived slug, which is fine for
       * a heading nothing points at.
       */
      const explicit = /\s*\{#([a-z0-9-]+)\}\s*$/.exec(heading[2]);
      const text = explicit ? heading[2].slice(0, explicit.index) : heading[2];
      const id = explicit
        ? explicit[1]
        : text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      out.push(`<h${level} id="${id}">${inline(text)}</h${level}>`);
      i++;
      continue;
    }

    if (line.startsWith("> ")) {
      const quote = [];
      while (i < lines.length && lines[i].startsWith("> ")) { quote.push(lines[i].slice(2)); i++; }
      out.push(`<blockquote><p>${inline(quote.join(" "))}</p></blockquote>`);
      continue;
    }

    // Tables: a header row, a separator, then body rows.
    if (line.startsWith("|") && /^\|[\s:|-]+\|$/.test(lines[i + 1] ?? "")) {
      const cells = (r) => r.split("|").slice(1, -1).map((c) => inline(c.trim()));
      const head = cells(line);
      i += 2;
      const body = [];
      while (i < lines.length && lines[i].startsWith("|")) { body.push(cells(lines[i])); i++; }
      out.push(
        `<table><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr>` +
          body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") +
          `</table>`,
      );
      continue;
    }

    const ordered = /^\d+\.\s+/.test(line);
    const bullet = /^[-*]\s+/.test(line);
    if (ordered || bullet) {
      const tag = ordered ? "ol" : "ul";
      const items = [];
      const re = ordered ? /^\d+\.\s+/ : /^[-*]\s+/;
      while (i < lines.length && re.test(lines[i])) {
        let text = lines[i].replace(re, "");
        i++;
        // Continuation lines are indented under their item.
        while (i < lines.length && /^\s{2,}\S/.test(lines[i])) { text += " " + lines[i].trim(); i++; }
        items.push(`<li>${inline(text)}</li>`);
      }
      out.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^[#>|`-]/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`);
    else i++;
  }
  if (inCode && code.length) flushCode();
  return out.join("\n");
}

const md = await readFile(join(root, "TUTORIAL.md"), "utf8");
const shell = await readFile(join(root, "docs", "_shell.html"), "utf8");
const body = render(md.replace(/^# .*\n/, ""));

await writeFile(
  join(root, "docs", "TUTORIAL.html"),
  shell.replace("<!--TITLE-->", "Tutorial — Slidecraft").replace("<!--BODY-->", body),
);
console.log("site: wrote docs/TUTORIAL.html");
