"""Check review sources, generated material and the compiled paper."""

import argparse
from bisect import bisect_right
import os
from pathlib import Path
import re
import sys
import shutil
import subprocess
import xml.etree.ElementTree as ET

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acl_resource.analysis.lint_release import local_identities, privacy_reasons, scan_text


COMMAND = re.compile(r"\\([a-zA-Z@]+)(\*)?")
FORMAT = re.compile(r"n|text|f[0-4]|a[0-4]|s?pts[0-2]|pct[0-2]|usd[2-4]|s[1-4]|p")
KEY = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+")
NUMBER_DEF = re.compile(r"\\expandafter\s*\\def\s*\\csname\s+paper@num@([^@\s]+)@([^\s\\]+)\s*\\endcsname")
MATH = re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]|"
                  r"\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|math|displaymath)\}"
                  r".*?\\end\{\1\}", re.S)
SECTION = {"section", "subsection", "subsubsection", "paragraph", "subparagraph", "chapter"}
CITE = re.compile(r"(?:[Cc]ite(?:t|p|alt|alp|author|year|yearpar|num|talias|palias)?|nocite)$")
REFERENCE = {"ref", "pageref", "autoref", "eqref"}
ANONYMOUS = {"", "anonymous", "anonymous acl submission"}


def mask_comments(source):
    """Return TeX with comments blanked, preserving line and column offsets."""
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", lambda m: m[1] + " " * (len(m[0]) - len(m[1])), source)


def line_number(source, offset):
    """Return the one-based line at a source offset."""
    return 1 + source.count("\n", 0, offset)


def group(source, start, opener, closer):
    """Return a balanced argument and its end offset, or None for a missing group."""
    while start < len(source) and source[start].isspace():
        start += 1
    if not source.startswith(opener, start):
        return None
    depth = 1
    pos = start + 1
    while pos < len(source):
        char = source[pos]
        if char == "\\":
            pos += 2
            continue
        if char == opener:
            depth += 1
        if char == closer:
            depth -= 1
            if depth == 0:
                return source[start + 1:pos], pos + 1
        pos += 1
    return None


def arguments(source, position):
    """Return optional arguments, the first required argument, and the end offset."""
    options = []
    while (option := group(source, position, "[", "]")) is not None:
        value, position = option
        options.append(value.strip())
    arg = group(source, position, "{", "}")
    if arg is None:
        return options, None, position
    return options, arg[0].strip(), arg[1]


def declaration_offsets(source):
    """Return offsets of control sequences at their declarations."""
    pattern = r"\\(?:(?:new|renew|provide)command\*?\s*\{?\s*|def\s*)(?=\\)"
    return {m.end() for m in re.finditer(pattern, source)}


def iter_numbers(source):
    """Yield (format, key, line) of each number call, raising on malformed calls."""
    source = mask_comments(source)
    declarations = declaration_offsets(source)
    for token in re.finditer(r"\\num(?![a-zA-Z@])", source):
        if token.start() in declarations:
            continue
        opts, key, _ = arguments(source, token.end())
        line = line_number(source, token.start())
        if len(opts) != 1 or not FORMAT.fullmatch(opts[0]) or not key or not KEY.fullmatch(key):
            raise ValueError(f"line {line}: invalid number format or key")
        yield opts[0], key, line


def prose_digits(source, ignored):
    """Yield the line of each prose digit sequence outside allowed command arguments and math indices."""
    for match in MATH.finditer(source):
        for index in re.finditer(r"(?<!\\)[_^]\s*", match[0]):
            pos = match.start() + index.end()
            arg = group(source, pos, "{", "}")
            ignored.append((pos, arg[1] if arg else pos + 1))
    chars = list(source)
    for start, end in ignored:
        chars[start:end] = [ch if ch == "\n" else " " for ch in chars[start:end]]
    for match in re.finditer(r"\d+", "".join(chars)):
        yield line_number(source, match.start())


def command_sites(source):
    """Yield commands and their balanced arguments in source order."""
    declarations = declaration_offsets(source)
    for match in COMMAND.finditer(source):
        if match.start() in declarations:
            continue
        opts, arg, end = arguments(source, match.end())
        yield match, opts, arg, end


def bibliography_path(paper, name):
    """Return the first matching BibTeX input on the supported bibliography search path."""
    search = [paper]
    for value in os.environ.get("BIBINPUTS", "").split(os.pathsep):
        if value:
            search.append(paper / value)
    search.append(paper / ".." / "related_work")
    name = name if name.endswith(".bib") else name + ".bib"
    return next((path.resolve() for folder in search if (path := folder / name).is_file()), None)


def source_order(paper, sources):
    """Return included files with the offset where appendix sectioning begins in each."""
    visited = {}
    in_appendix = False

    def visit(path):
        nonlocal in_appendix
        if path in visited or path not in sources:
            return
        source = sources[path]
        visited[path] = 0 if in_appendix else len(source)
        for token, _, arg, end in command_sites(source):
            if token[1] == "appendix":
                in_appendix = True
            elif token[1] in ("include", "input") and arg:
                child = paper / (arg if arg.endswith(".tex") else arg + ".tex")
                visit(child.resolve())
            if in_appendix:
                visited[path] = min(visited[path], end)

    visit(paper / "main.tex")
    return visited


def display_path(path, paper):
    """Return a relative filename for a diagnostic."""
    return Path(os.path.relpath(path, paper)).as_posix()


def source_privacy(paper, paths, terms):
    """Return privacy errors and unavailable-tool notices for paper inputs."""
    errors, notices = [], []
    identities = local_identities()
    for path in sorted(paths):
        display = display_path(path, paper)
        if path.suffix.lower() == ".pdf":
            pdf_errors, pdf_notices = check_metadata(path, display, identities, terms)
            errors.extend(pdf_errors)
            notices.extend(pdf_notices)
            if shutil.which("pdftotext"):
                try:
                    for page, content in enumerate(poppler_text(path), 1):
                        for entry in content.splitlines():
                            for reason in privacy_reasons(entry, identities, terms):
                                errors.append(f"{display}:{page}: {reason}")
                except (OSError, ValueError):
                    errors.append(f"{display}:1: unreadable generated figure text")
            else:
                notices.append(f"{display}:1: pdftotext unavailable, generated figure text unchecked")
        else:
            errors.extend(scan_text(path, display, identities, terms))
    return errors, notices


def check_calls(source, display, definitions):
    """Yield diagnostics for missing or malformed number calls."""
    try:
        for fmt, key, line in iter_numbers(source):
            if (fmt, key) not in definitions:
                yield f"{display}:{line}: undefined number format/key"
    except ValueError as exc:
        yield f"{display}:{str(exc).removeprefix('line ')}"


def located_commands(source, display):
    """Yield parsed commands with relative file and line information."""
    newlines = [m.start() for m in re.finditer("\n", source)]
    for token, opts, arg, end in command_sites(source):
        line = bisect_right(newlines, token.start()) + 1
        yield token, opts, arg, end, f"{display}:{line}"


def check_prose(source, commands, appendix, display, terms):
    """Return typed-digit errors and allowed literals in a section."""
    ignored, notices = [], []
    for token, _, arg, end, location in commands:
        cmd = token[1]
        if cmd == "literal":
            values = re.findall(r"[+-]?\d+(?:[,.]\d+)*", arg or "")
            safe = [v for v in values if not any(t in v for t in terms)]
            notices.append(f"{location}: literal " + ", ".join(safe))
        if (cmd in {"num", "label", "input", "includegraphics", "literal"} | REFERENCE or
                CITE.fullmatch(cmd) or cmd == "citetext" or (cmd in SECTION and token.start() >= appendix)):
            ignored.append((token.start(), end))
    errors = [f"{display}:{line}: typed digit in prose" for line in sorted(set(prose_digits(source, ignored)))]
    return errors, notices


def check_review(commands, active, main, numbers, submission):
    """Yield diagnostics for review mode, anonymity, required headings and submission flags."""
    style_location = "main.tex:1"
    reviewed = limitations = False
    for path, sites in commands.items():
        for token, opts, arg, _, location in sites:
            cmd = token[1]
            if cmd == "author" and (arg is None or arg.casefold() not in ANONYMOUS):
                yield f"{location}: named author"
            if (cmd in {"acknowledgments", "acknowledgements", "acks"} or
                    cmd in SECTION | {"begin"} and re.search(r"\backnowledg(?:e)?ments?\b", arg or "", re.I)):
                yield f"{location}: acknowledgements in review paper"
            if cmd == "section" and token[2] and arg == "Limitations" and path in active:
                limitations = True
            if cmd == "usepackage" and path == main and "acl" in [s.strip() for s in (arg or "").split(",")]:
                options = {opt.strip() for part in opts for opt in part.split(",")}
                reviewed = "review" in options and not options.intersection({"final", "preprint"})
                style_location = location
            if cmd == "placeholder" and submission:
                yield f"{location}: leftover placeholder"
            if cmd == "papersynthetictrue" and path == numbers and submission:
                yield f"{location}: made-up data flag"
    if not reviewed:
        yield f"{style_location}: ACL review style required"
    if not limitations:
        yield "main.tex:1: unnumbered Limitations section missing"


def check_references(paper, commands, active):
    """Return unresolved reference and citation errors and the used bibliography files."""
    labels, cited, bibs, refs, errors = set(), [], set(), [], []
    for path, sites in commands.items():
        for token, _, arg, _, location in sites:
            cmd = token[1]
            if cmd == "label" and path in active:
                labels.add(arg)
            if cmd in REFERENCE:
                refs.append((arg, location))
            if CITE.fullmatch(cmd):
                cited.extend((key.strip(), location) for key in (arg or "").split(",")
                             if cmd != "nocite" or key.strip() != "*")
            if cmd == "bibliography" and path in active:
                for name in (arg or "").split(","):
                    bib = bibliography_path(paper, name.strip())
                    if bib:
                        bibs.add(bib)
                    else:
                        errors.append(f"{location}: missing bibliography file")
    bib_keys = set()
    for path in bibs:
        content = mask_comments(path.read_text(encoding="utf-8"))
        bib_keys.update(re.findall(r"(?im)^\s*@\w+\s*[{(]\s*([^,\s]+)\s*,", content))
    errors.extend(f"{where}: undefined reference" for key, where in refs if key not in labels)
    errors.extend(f"{where}: undefined citation" for key, where in cited if key not in bib_keys)
    return errors, bibs


def check_sources(paper, submission=False, terms=()):
    """Return error and notice lists from source and generated-file checks."""
    paper = Path(paper).resolve()
    terms = tuple(term.casefold() for term in terms if term)
    errors, notices, sources, commands = [], [], {}, {}
    generated = paper / "generated"
    main, numbers = paper / "main.tex", generated / "numbers.tex"
    tex_files = set(paper.glob("*.tex")) | set((paper / "sections").rglob("*.tex")) | set(generated.rglob("*.tex"))
    for path in sorted(tex_files | {main, numbers}):
        display = display_path(path, paper)
        try:
            sources[path] = mask_comments(path.read_text(encoding="utf-8"))
            commands[path] = list(located_commands(sources[path], display))
        except (OSError, UnicodeError):
            errors.append(f"{display}:1: missing or unreadable TeX source")
    active = source_order(paper, sources)
    errors.extend(check_review(commands, active, main, numbers, submission))
    definitions = set(NUMBER_DEF.findall(sources.get(numbers, "")))
    for path, source in sources.items():
        display = display_path(path, paper)
        if path != numbers:
            errors.extend(check_calls(source, display, definitions))
        if (paper / "sections") in path.parents:
            typed, literals = check_prose(source, commands[path], active.get(path, len(source)), display, terms)
            errors.extend(typed)
            notices.extend(literals)
    missing, bibs = check_references(paper, commands, active)
    errors.extend(missing)
    paths = tex_files | bibs | set(paper.glob("*.bib")) | set(paper.glob("*.sty")) | set(paper.glob("*.cls"))
    paths.update(path for path in generated.rglob("*") if path.is_file())
    if (paper / "static_figures").is_dir():          # the hand-drawn figures, checked like the generated ones
        paths.update(path for path in (paper / "static_figures").rglob("*") if path.is_file())
    if (paper / "latexmkrc").exists():
        paths.add(paper / "latexmkrc")
    private, unavailable = source_privacy(paper, paths, terms)
    return errors + private, notices + unavailable


def poppler(command, args, path):
    """Return a PDF utility's stdout, or raise an error without disclosing its inputs."""
    proc = subprocess.run([command, *args, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace", env={**os.environ, "LC_ALL": "C"})
    if proc.returncode:
        raise ValueError(f"{command} could not inspect PDF")
    return proc.stdout


def check_metadata(path, display, identities, terms, info=None):
    """Return privacy errors and tool-availability notices for PDF metadata."""
    if not shutil.which("pdfinfo"):
        return [], [f"{display}:1: pdfinfo unavailable, metadata unchecked"]
    errors = []
    try:
        if info is None:
            info = poppler("pdfinfo", [], path)
        for line, entry in enumerate(info.splitlines(), 1):
            key, sep, value = entry.partition(":")
            if not sep:
                continue
            if key == "Author" and value.strip().casefold() not in ANONYMOUS:
                errors.append(f"{display}:{line}: named PDF author")
            for reason in privacy_reasons(value, identities, terms):
                errors.append(f"{display}:{line}: PDF metadata contains {reason}")
        xml = poppler("pdfinfo", ["-meta"], path)
        if xml.strip():
            root = ET.fromstring(xml.strip())
            for node in root.iter():
                if node.tag.endswith("}creator"):
                    author = " ".join(" ".join(node.itertext()).split()).casefold()
                    if author not in ANONYMOUS:
                        errors.append(f"{display}:1: named PDF metadata creator")
                for value in (node.text or "", *node.attrib.values()):
                    for reason in privacy_reasons(value, identities, terms):
                        errors.append(f"{display}:1: PDF metadata contains {reason}")
    except (ValueError, ET.ParseError):
        errors.append(f"{display}:1: unreadable PDF metadata")
    return errors, []


def check_pdf(path, terms=(), display=None):
    """Return errors and availability notices for page sizes, page count, fonts and metadata."""
    path = Path(path)
    display = display or path.name
    errors, notices = [], []
    if not path.is_file():
        return [f"{display}:1: compiled PDF missing"], notices
    for tool in ("pdfinfo", "pdftotext", "pdffonts"):
        if not shutil.which(tool):
            notices.append(f"{display}:1: {tool} unavailable, dependent checks skipped")
            continue
        try:
            if tool == "pdfinfo":
                info = poppler(tool, [], path)
                count = re.search(r"(?m)^Pages:\s*(\d+)", info)
                if not count:
                    raise ValueError
                pages = int(count[1])
                detail = poppler(tool, ["-f", "1", "-l", str(pages)], path)
                sizes = re.findall(r"(?m)^Page\s+(\d+)\s+size:\s*(.*)$", detail)
                if len(sizes) != pages:
                    raise ValueError
                for page, size in sizes:
                    if "(A4)" not in size:
                        errors.append(f"{display}:{page}: page size must be A4")
                private, skipped = check_metadata(path, display, local_identities(), terms, info)
                errors.extend(private)
                notices.extend(skipped)
            elif tool == "pdftotext":
                pages = poppler_text(path)
                heading = re.compile(r"(?im)^\s*(?:\d+[ \t]+)?Limitations(?:[ \t]+\d+)?\s*$")
                start = next((i for i, page in enumerate(pages) if heading.search(page)), None)
                if start is None:
                    errors.append(f"{display}:1: Limitations heading missing from PDF")
                elif start > 8:
                    errors.append(f"{display}:{start + 1}: more than eight pages before Limitations")
            else:
                listing = poppler(tool, [], path)
                for line, font in enumerate(listing.splitlines()[2:], 3):
                    if re.search(r"\bType\s+3\b", font):
                        errors.append(f"{display}:{line}: Type 3 font")
                    fields = font.split()
                    if len(fields) < 5 or fields[-5] != "yes":
                        errors.append(f"{display}:{line}: font not embedded")
        except (OSError, ValueError):
            errors.append(f"{display}:1: {tool} could not verify PDF")
    return errors, notices


def poppler_text(path):
    """Return extracted text split at physical PDF page boundaries."""
    proc = subprocess.run(["pdftotext", "-raw", str(path), "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode:
        raise ValueError("could not extract text")
    return proc.stdout.split("\f")


def main(argv=None):
    """Return the paper lint exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path(__file__).resolve().parents[1] / "paper")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--terms", type=Path)
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--sources-only", action="store_true", help="check draft sources without a compiled PDF")
    args = parser.parse_args(argv)
    if args.sources_only and args.submission:
        parser.error("submission requires a PDF check")
    try:
        terms = [] if args.terms is None else [s.strip().casefold() for s in args.terms.read_text().splitlines() if s.strip()]
        errors, notices = check_sources(args.paper, args.submission, terms)
        if not args.sources_only:
            pdf = args.pdf or args.paper / "build" / "main.pdf"
            pdf_errors, pdf_notices = check_pdf(pdf, terms, display_path(pdf, args.paper))
            errors.extend(pdf_errors)
            notices.extend(pdf_notices)
    except (OSError, UnicodeError):
        print("paper:1: unreadable input")
        return 1
    for message in notices + errors:
        print(message)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
