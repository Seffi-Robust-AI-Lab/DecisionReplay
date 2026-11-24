"""Inspect the text, file inventory and public Parquet tables of a release."""

import argparse
import codecs
import json
import os
from pathlib import Path
import re
import sys
import socket

import pyarrow as pa
import pyarrow.parquet as pq

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acl_resource.data.schema import (QUESTION_TYPES, SCHEMA_VERSION, TABLES, TABLES_BY_TIER, TRANSPORTS,
                                      corpus_tables, usage_schema)


TOOL_FILES = {".git", ".DS_Store", "__pycache__", "ruvector.db", "agentdb.rvf", "agentdb.rvf.lock"}
EMAIL = re.compile(r"(?<![\w.%+@\-])[\w.%+\-]+@[\w-]+(?:\.[\w-]+)+")
WEB_URL = re.compile(r"\b(?:https?|ftp)://[^\s{}<>\"']+", re.I)
ABS_PATH = re.compile(r"(?:^|[\s({\[=:\"'`,;])(?:/{1,2}[\w.~-]|[a-z]:[\\/]|\\\\[\w.-]+\\[\w.-]+(?=[\\\s\x22\x27)}\],;]|$))", re.I)
ACCOUNT = re.compile(r"(?:\b(?:user(?:name)?|login|host(?:name)?)\s*[\x22']?\s*[:=]\s*[\x22']?"
                     r"[\w][\w.-]*|(?<![\w.@\\])@[\w-]+\b|(?<![\w/])~[\w.-]+/)", re.I)
PRIVATE_HOST = re.compile(r"\b[\w-]+\.(?:local|lan|internal|localdomain)\b", re.I)
HASH = re.compile(r"[0-9a-fA-F]{64}")


def local_identities():
    """Return exact-word patterns for the current account and machine identifiers."""
    machine = socket.gethostname()
    names = {os.environ.get(k) for k in ("USER", "LOGNAME", "USERNAME", "HOSTNAME", "COMPUTERNAME")}
    names.update((machine, machine.partition(".")[0]))
    return tuple(re.compile(r"(?<![\w-])" + re.escape(s) + r"(?![\w-])", re.I)
                 for s in sorted(s for s in names if s))


def privacy_reasons(line, identities, terms=()):
    """Return categories of identifying information without repeating the matched text."""
    reasons = []
    if EMAIL.search(line):
        reasons.append("e-mail address")
    # /dev/null is the null device of a shell redirection, not a location on somebody's machine
    if ABS_PATH.search(WEB_URL.sub(" ", line).replace("/dev/null", " ")) or re.search(r"\bfile:/", line, re.I):
        reasons.append("absolute path")
    account_text = re.sub(r"^\s*@\w+\s*[({]", " ", line)
    if ACCOUNT.search(account_text) or PRIVATE_HOST.search(line) or any(p.search(line) for p in identities):
        reasons.append("user name or host name")
    if any(term in line.casefold() for term in terms):
        reasons.append("local anonymity term")
    return reasons


def text_encoding(path):
    """Return a text encoding for an ordinary text file, or None for a binary file."""
    with path.open("rb") as stream:
        prefix = stream.read(4096)
    for signature, encoding in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                                (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if prefix.startswith(signature):
            return encoding
    if b"\x00" in prefix or prefix.startswith((b"%PDF-", b"PAR1")):
        return None
    if any(ch < 32 and ch not in b"\t\n\r\f" for ch in prefix):
        return None
    return "utf-8-sig"


def scan_text(path, display, identities, terms=()):
    """Yield file and line diagnostics for identifying information in text."""
    encoding = text_encoding(path)
    if encoding:
        with path.open(encoding=encoding, errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                if path.suffix == ".py":
                    line = re.sub(r"^(\s*)@(?=[a-zA-Z_]\w*(?:\.\w+)*(?:\s*\(|\s*$))", r"\1 ", line)
                for reason in privacy_reasons(line, identities, terms):
                    yield f"{display}:{number}: {reason}"


def contains_strings(kind):
    """Return whether an Arrow type can contain string values."""
    if pa.types.is_dictionary(kind):
        return contains_strings(kind.value_type)
    return (pa.types.is_string(kind) or pa.types.is_large_string(kind) or pa.types.is_string_view(kind)
            or any(contains_strings(kind.field(i).type) for i in range(kind.num_fields)))


def string_values(value):
    """Yield string values from a Parquet cell, including lists, maps and structs."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple, dict)):
        for child in value.values() if isinstance(value, dict) else value:
            yield from string_values(child)


def check_table(path, display, tier, providers, version=SCHEMA_VERSION):
    """Yield tier violations and the first offending row of each string column.

    ``version`` is the schema version of the corpus: the open tier of version 2 carries usage and latency, and its
    usage table has the request totals only (no serving host, transport or attempts).
    """
    tables = corpus_tables(version, tier)
    if path.stem not in tables:
        yield f"{display}:1: table forbidden in {tier} tier"
    allowed = set(QUESTION_TYPES) | set(TRANSPORTS) | providers
    try:
        parquet = pq.ParquetFile(path)
        fields = set(parquet.schema_arrow.names)
        if path.stem in tables:
            for table in TABLES_BY_TIER["local"]:
                if table not in tables and set(TABLES[table].names) <= fields:
                    yield f"{display}:1: contents of {table} table forbidden in {tier} tier"
            if path.stem == "usage" and fields != set(usage_schema(version).names):
                yield f"{display}:1: usage columns differ from the schema of the {tier} tier"
        columns = [field.name for field in parquet.schema_arrow if contains_strings(field.type)]
        if not columns:
            return
        first = 1
        reported = set()
        for batch in parquet.iter_batches(columns=columns):
            for field, values in zip(batch.schema, batch.columns):
                if field.name in reported:
                    continue
                for offset, cell in enumerate(values.to_pylist()):
                    if any(s not in allowed and not HASH.fullmatch(s) for s in string_values(cell)):
                        yield f"{display}:{first + offset}: free-text string in {tier} table"
                        reported.add(field.name)
                        break
            if len(reported) == len(columns):
                break
            first += len(batch)
    except (ValueError, OSError, pa.ArrowException):
        yield f"{display}:1: unreadable Parquet table"


def public_tier(folder, expected):
    """Return the public tier, its provider list and its schema version, or None for other manifests."""
    manifest = folder / "manifest.json"
    if manifest.is_symlink():
        raise ValueError
    data = json.loads(manifest.read_bytes())
    tier = data.get("tier") if isinstance(data, dict) else None
    if expected and expected != tier:
        raise ValueError
    if tier in ("gated", "open"):
        providers = data.get("providers", [])
        if not isinstance(providers, list) or not all(isinstance(s, str) and s for s in providers):
            raise ValueError
        version = data.get("schema_version", SCHEMA_VERSION)
        corpus_tables(version, tier)                  # raises on an unsupported version
        return tier, set(providers), version
    return None


def parquet_payload(path):
    """Return whether a file has the Parquet header, including renamed files."""
    with path.open("rb") as stream:
        return stream.read(4) == b"PAR1"


def check_release(directory):
    """Return release errors and byte totals for each top-level folder."""
    directory = Path(directory)
    errors, sizes, restricted = [], {}, {}
    if not directory.is_dir():
        return ["release:1: directory missing"], sizes
    identities = local_identities()
    for folder, dirs, files in os.walk(directory, onerror=lambda exc: errors.append("release:1: unreadable directory")):
        folder = Path(folder)
        relative = folder.relative_to(directory)
        dirs.sort()
        files.sort()
        expected = folder.name if folder.parent.name == "processed" and folder.name in ("gated", "open") else None
        manifest = folder / "manifest.json"
        if expected or manifest.name in files:
            display = relative / manifest.name
            try:
                public = public_tier(folder, expected)
                if public is not None:
                    restricted[folder] = public
            except (ValueError, OSError):
                errors.append(f"{display}:1: unreadable or inconsistent manifest")
        for name in dirs.copy() + files:
            path = folder / name
            display = relative / name
            if name in TOOL_FILES or name.endswith(".pyc"):
                errors.append(f"{display}:1: tool or system file")
            if path.is_symlink():
                errors.append(f"{display}:1: cannot verify symbolic link")
                if name in dirs:
                    dirs.remove(name)
                continue
            if name in dirs:
                sizes.setdefault(display.parts[0], 0)
                continue
            top = display.parts[0] if len(display.parts) > 1 else "."
            try:
                sizes[top] = sizes.get(top, 0) + path.stat().st_size
                public = next((restricted[parent] for parent in path.parents if parent in restricted), None)
                if public and (path.suffix.lower() == ".parquet" or parquet_payload(path)):
                    errors.extend(check_table(path, display, *public))
                elif path.suffix.lower() != ".parquet":
                    errors.extend(scan_text(path, display, identities))
            except OSError:
                errors.append(f"{display}:1: unreadable file")
    return errors, sizes


def main(argv=None):
    """Return the release lint exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    errors, sizes = check_release(args.directory)
    for name, size in sorted(sizes.items()):
        print(f"{name}: {size:,} bytes")
    for message in errors:
        print(message)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
