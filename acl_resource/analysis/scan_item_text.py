"""Check that a package holds no text of the benchmark rows.

    python3 -B -m acl_resource.analysis.scan_item_text PACKAGE_DIR ROWS_DIR

Every sequence of eight words (letters only, lower case) of the package's text files and PDFs is looked up in the
local benchmark rows.  The output gives counts per package file and one example taken from the package side; it
never prints text of the rows.  A hit is a reason to look, not a verdict: runs of one repeated word and code
idioms occur on both sides.
"""
import gzip, re, subprocess, sys
from collections import Counter
from pathlib import Path

package, rows_dir = Path(sys.argv[1]), Path(sys.argv[2])
WORD = re.compile(r"[a-z]{2,}")
N = 8
grams = {}
for path in sorted(p for p in package.rglob("*") if p.is_file()):
    if path.suffix == ".parquet":
        continue
    if path.suffix == ".pdf":
        text = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, errors="replace").stdout
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    words = WORD.findall(text.lower())
    for i in range(len(words) - N + 1):
        grams.setdefault(" ".join(words[i:i + N]), path.relative_to(package).as_posix())
print(len(grams), "eight-word sequences in the package's text")
hits, lines = Counter(), 0
example = {}
for path in sorted(rows_dir.glob("*.jsonl.gz")):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            words = WORD.findall(line.lower())
            for i in range(len(words) - N + 1):
                g = " ".join(words[i:i + N])
                if g in grams:
                    hits[grams[g]] += 1
                    example.setdefault(grams[g], g)
print(lines, "rows read;", sum(hits.values()), "shared sequences in", len(hits), "package files")
for name, count in hits.most_common(15):
    print(count, name, "| package-side example:", example[name])
