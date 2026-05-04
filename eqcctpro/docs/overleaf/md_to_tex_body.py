#!/usr/bin/env python3
"""One-off: convert EQCCTPro_Draft.md lines 7-252 to LaTeX for body.tex."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "EQCCTPro_Draft.md"
OUT = Path(__file__).resolve().parent / "body.tex"


def escape_tex(s: str) -> str:
    # Order matters
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    return s


def inline_format(s: str) -> str:
    """Apply **bold**, *italic*, `code` after escaping."""
    # `code`
    def repl_code(m):
        return "\\texttt{" + escape_tex(m.group(1)) + "}"

    s = re.sub(r"`([^`]+)`", repl_code, s)

    # **bold**
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: "\\textbf{" + escape_tex(m.group(1)) + "}", s)

    # *italic* (single asterisks, not double)
    s = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        lambda m: "\\emph{" + escape_tex(m.group(1)) + "}",
        s,
    )

    # Remaining escape for plain text parts - split by latex commands is hard; escape rest
    # Simpler: escape whole line then un-escape command prefixes - skip, do char escape on full line first for non-markdown
    return s


def process_line(line: str) -> str | None:
    line = line.rstrip()
    if not line.strip():
        return ""

    # Headers
    m = re.match(r"^(#{1,4})\s+(.*)$", line)
    if m:
        level = len(m.group(1))
        title = m.group(2).strip()
        title = inline_format(title)
        if level == 1:
            return None  # skip duplicate title
        if level == 2:
            return "\\section{" + title + "}"
        if level == 3:
            return "\\subsection{" + title + "}"
        if level == 4:
            return "\\subsubsection{" + title + "}"

    # Numbered list
    m = re.match(r"^(\d+)\.\s+(.*)$", line)
    if m:
        body = inline_format(escape_tex(m.group(2)))
        return f"\\item {body}"

    if line.strip() == "\\newpage":
        return "\\clearpage"

    # Continuation of itemize
    if line.startswith("(") and ")" in line[:4]:
        return inline_format(escape_tex(line))

    body = inline_format(escape_tex(line))
    return body


def main():
    text = MD.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Lines 7-252 (1-based) -> index 6:252
    chunk = lines[6:252]
    out_lines = []
    in_list = False
    for line in chunk:
        raw = line
        if raw.strip().startswith("1. ") and not in_list:
            out_lines.append("\\begin{enumerate}")
            in_list = True
        elif in_list and raw.strip() and not re.match(r"^\d+\.\s", raw) and not raw.strip().startswith("\\newpage"):
            if raw.strip().startswith("These models"):
                out_lines.append("\\end{enumerate}")
                in_list = False

        proc = process_line(raw)
        if proc is None:
            continue
        if proc == "":
            out_lines.append("")
            continue
        out_lines.append(proc)

    if in_list:
        out_lines.append("\\end{enumerate}")

    OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
