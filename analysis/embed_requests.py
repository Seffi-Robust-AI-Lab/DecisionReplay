"""Sentence embeddings of the request text, for the text variant of the router.  A local step.

    python3 -B -m analysis.embed_requests --rows DIR --corpus DIR --out DIR

It reads the benchmark rows (text), so it runs only where they are, with a sentence encoder that is already in
the local model cache (no download).  The text of a request is its state, serialised as JSON, followed by the
instructions of its questions; the encoder truncates it at its maximum length.  The output, one float32 matrix
per benchmark, stays local: it is derived from benchmark text.
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

from data import tables as T
from data.load import corpus_rows, list_benchmarks

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def request_text(row):
    state = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
    return state + "\n" + "\n".join(str(q.get("instructions") or "") for q in row["questions"].values())


def run(rows_dir, corpus, out, model=MODEL, batch=128):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0)
    encoder = SentenceTransformer(model, device="cpu")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = dict(model=model, max_tokens=int(encoder.max_seq_length), normalised=True, device="cpu",
                text="state as JSON, then the instructions of the questions", benchmarks={})
    for b in list_benchmarks(corpus):
        cid = b["catalog_id"]
        rows = corpus_rows(corpus, rows_dir, cid)
        if len(rows) != int(b["requests"]):
            raise ValueError(f"benchmark {cid}: rows do not match the corpus")
        x = encoder.encode([request_text(r) for r in rows], batch_size=batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(out / f"catalog-{cid:03d}.npy", x)
        info["benchmarks"][str(cid)] = dict(requests=int(x.shape[0]), dimensions=int(x.shape[1]))
        print(f"  benchmark {cid:3d}: {x.shape}", flush=True)
    (out / "features.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Embed request text for the text variant of the router")
    for name in ("rows", "corpus", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)
    run(a.rows, a.corpus, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
