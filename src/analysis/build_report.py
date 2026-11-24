"""Build report/report_draft.md from report_template.md, replacing each {{include: path}} line with that file."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
INCLUDE = re.compile(r"^\{\{include:\s*(.+?)\s*\}\}\s*$")


def include(rel: str) -> str:
    path = (REPORT / rel) if not rel.startswith("/") else ROOT / rel.lstrip("/")
    if not path.exists():
        raise FileNotFoundError(f"include not found: {rel}")
    text = path.read_text()
    if path.suffix != ".md":  # e.g. env.txt: show verbatim
        return "```text\n" + text.rstrip() + "\n```"
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    out = []
    for line in lines:
        if line.startswith("## "):
            line = "#### " + line[3:]
        out.append(line)
    return "\n".join(out).strip()


def main() -> None:
    template = (REPORT / "report_template.md").read_text().splitlines()
    out, n = [], 0
    for line in template:
        m = INCLUDE.match(line)
        if m:
            out.append(include(m.group(1)))
            n += 1
        else:
            out.append(line)
    draft = REPORT / "report_draft.md"
    draft.write_text("<!-- Generated from report_template.md by `python -m src.analysis.build_report`. "
                     "Edit the template, not this file. -->\n\n" + "\n".join(out) + "\n")
    missing = [p for p in re.findall(r"\]\(((?:figures|tables)/[^)]+)\)", draft.read_text())
               if not (REPORT / p).exists()]
    todos = draft.read_text().count("TODO(user)")
    print(f"wrote {draft}: {n} includes, {todos} TODO(user) markers, missing links: {missing or 'none'}")


if __name__ == "__main__":
    main()
