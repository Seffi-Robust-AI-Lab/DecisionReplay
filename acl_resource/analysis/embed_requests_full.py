"""Sentence embeddings of the whole request, answer options included.  A local step, like ``embed_requests``.

    python3 -B -m acl_resource.analysis.embed_requests_full --rows DIR --corpus DIR --out DIR [--device cpu]

``embed_requests`` embeds the state of a request and the instructions of its questions, cut at the encoder's
maximum length.  That text leaves out the answer options, and on long requests the cut removes what differs
between requests; on three benchmarks every request then has the same vector.  This module builds the
representation that a fair test of "richer request representations" needs.  The design was fixed before the
first run:

* text: the state, serialised as JSON as in ``embed_requests``; then, for every question in order, its
  instructions and its answer options (the ``criteria`` of the question, serialised as JSON in their order);
* no cut: the text is tokenised by the encoder's own tokeniser and split into consecutive windows of the
  encoder's maximum length (less its two special tokens); every window is embedded;
* one vector per request: the mean of its window vectors, weighted by their numbers of tokens, scaled to unit
  length.

It uses the sentence encoder of ``embed_requests`` from the local model cache (no download) and calls no other
model.  The first run used two CPU processes; on a loaded machine they were slow, so they were stopped after 13
benchmarks.  Four further benchmarks were embedded on the GPU (``--device mps``) until it ran out of the memory
that other applications share with it, and the last 19 on the CPU again, in one process.  On made-up text the
two devices give vectors with a cosine of at least 0.9999998.  ``--merge`` gathers the runs and records the device of every
benchmark; a router is fitted inside one benchmark at a time, so no benchmark mixes devices.  The output, one float32 matrix per benchmark and ``features.json``, is derived from benchmark text and
stays local; write it to a folder that is not versioned and is not the folder of the stored features, so that
the frozen results keep their inputs.  Added after the first run, on 23 November 2026, in answer to a review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from ..data import tables as T
from ..data.load import corpus_rows, list_benchmarks
from .embed_requests import MODEL

TEXT = "state as JSON; then for every question its instructions and its answer options as JSON; no cut"


def question_text(question):
    """Instructions and answer options of one question."""
    instructions = question.get("instructions")
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False, sort_keys=True) if instructions else ""
    criteria = question.get("criteria")
    return instructions + "\n" + (json.dumps(criteria, ensure_ascii=False) if criteria else "")


def full_text(row):
    """The whole text of a request: state, then instructions and options of every question."""
    state = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
    return state + "\n" + "\n".join(question_text(q) for q in row["questions"].values())


def windows(tokens, size):
    """[(start, end)] consecutive windows of at most ``size`` tokens; one empty window for an empty text."""
    if size < 1:
        raise ValueError("a window needs at least one token")
    if tokens == 0:
        return [(0, 0)]
    return [(start, min(start + size, tokens)) for start in range(0, tokens, size)]


def pool_windows(vectors, counts, owner, requests):
    """[requests, d] float32: token-weighted mean of the window vectors of every request, at unit length."""
    weights = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    out = np.zeros((requests, vectors.shape[1]))
    np.add.at(out, owner, vectors.astype(np.float64) * weights[:, None])
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(norm == 0, 1.0, norm)).astype(np.float32)


def embed_benchmark(encoder, texts, batch):
    """(matrix, number of windows, number of tokens) of the texts of one benchmark."""
    tokeniser = encoder.tokenizer
    size = int(encoder.max_seq_length) - 2
    ids = tokeniser(texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    chunks, counts, owner = [], [], []
    for request, sequence in enumerate(ids):
        for start, end in windows(len(sequence), size):
            chunks.append(tokeniser.decode(sequence[start:end]))
            counts.append(end - start)
            owner.append(request)
    vectors = encoder.encode(chunks, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False)
    return pool_windows(vectors, counts, np.asarray(owner), len(texts)), len(chunks), int(sum(counts))


def run(rows_dir, corpus, out, model=MODEL, batch=128, device="cpu", cids=None):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0)
    encoder = SentenceTransformer(model, device=device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = dict(model=model, max_tokens=int(encoder.max_seq_length), window=int(encoder.max_seq_length) - 2,
                normalised=True, device=device, text=TEXT, benchmarks={})
    for b in list_benchmarks(corpus):
        cid = b["catalog_id"]
        if cids and cid not in cids:
            continue
        rows = corpus_rows(corpus, rows_dir, cid)
        if len(rows) != int(b["requests"]):
            raise ValueError(f"benchmark {cid}: rows do not match the corpus")
        x, chunks, tokens = embed_benchmark(encoder, [full_text(r) for r in rows], batch)
        if not np.isfinite(x).all():
            raise ValueError(f"benchmark {cid}: a vector is not finite")
        np.save(out / f"catalog-{cid:03d}.npy", x)
        info["benchmarks"][str(cid)] = dict(requests=int(x.shape[0]), dimensions=int(x.shape[1]), windows=chunks,
                                            tokens=tokens, distinct=int(len(np.unique(x, axis=0))))
        print(f"  benchmark {cid:3d}: {x.shape}, {chunks} windows, {info['benchmarks'][str(cid)]['distinct']} "
              f"distinct vectors", flush=True)
    (out / "features.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return info


def merge(parts, rows_dir, corpus, out, model=MODEL):
    """Gather the matrices of several runs (``parts``: [(folder, device)]) into one folder with one features.json.

    The window and token counts are recounted with the encoder's tokeniser, so that a run that was stopped before
    it wrote its features.json loses nothing; no model is run.  Every benchmark of the corpus must be present
    exactly once.
    """
    import shutil
    from transformers import AutoTokenizer
    tokeniser = AutoTokenizer.from_pretrained(model)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    found = {}
    for folder, device in parts:
        for path in sorted(Path(folder).glob("catalog-*.npy")):
            cid = int(path.stem.split("-")[1])
            if cid in found:
                raise ValueError(f"benchmark {cid} is in two runs")
            found[cid] = (path, device)
    missing = [b["catalog_id"] for b in list_benchmarks(corpus) if b["catalog_id"] not in found]
    if missing:
        raise ValueError(f"benchmark {missing[0]} has no matrix ({len(missing)} missing)")
    size = None
    info = dict(model=model, normalised=True, text=TEXT, benchmarks={})
    for b in list_benchmarks(corpus):
        cid = b["catalog_id"]
        if cid not in found:
            raise ValueError(f"benchmark {cid} has no matrix")
        path, device = found[cid]
        x = np.load(path, allow_pickle=False)
        if x.shape[0] != int(b["requests"]) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"benchmark {cid}: the matrix does not match the corpus")
        if size is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model, device="cpu")
            size = int(encoder.max_seq_length) - 2
            info.update(max_tokens=int(encoder.max_seq_length), window=size)
            del encoder
        rows = corpus_rows(corpus, rows_dir, cid)
        if len(rows) != x.shape[0]:
            raise ValueError(f"benchmark {cid}: rows do not match the matrix")
        ids = tokeniser([full_text(r) for r in rows], add_special_tokens=False, truncation=False,
                        verbose=False)["input_ids"]
        shutil.copyfile(path, out / path.name)
        info["benchmarks"][str(cid)] = dict(
            requests=int(x.shape[0]), dimensions=int(x.shape[1]), device=device,
            windows=int(sum(len(windows(len(s), size)) for s in ids)), tokens=int(sum(len(s) for s in ids)),
            distinct=int(len(np.unique(x, axis=0))))
    (out / "features.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Embed the whole request, answer options included")
    for name in ("rows", "corpus", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cids", type=int, nargs="*")
    ap.add_argument("--merge", nargs="+", metavar="FOLDER=DEVICE",
                    help="gather finished runs into --out instead of embedding")
    a = ap.parse_args(argv)
    if a.merge:
        merge([tuple(item.split("=", 1)) for item in a.merge], a.rows, a.corpus, a.out)
    else:
        run(a.rows, a.corpus, a.out, device=a.device, cids=a.cids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
