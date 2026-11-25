"""Further representations of the request text for the router.  A local step, like ``embed_requests``.

    python3 -B -m analysis.featurize_requests --kind tfidf|lm --rows DIR --corpus DIR --out DIR

It reads the benchmark rows (text), so it runs only where they are, and it writes one float32 matrix per
benchmark, which stays local because it is derived from benchmark text.  The text of a request is the one the
sentence-encoder variant uses (``embed_requests.request_text``).

* ``tfidf``: word and bigram TF-IDF of the request text, reduced to at most 128 dimensions by a truncated SVD and
  scaled to unit length.  Both transforms are fitted on all requests of the benchmark; they read no label.
* ``lm``: the mean of the last hidden states of a small decoder language model from the local model cache
  (no download), over the first 256 tokens of the text, reduced per benchmark to at most 128 dimensions by PCA
  and scaled to unit length.  The unreduced means are kept beside the reduced ones.

``features.json`` in the output folder records the model, its cache revision, the truncation, the device and a
checksum of every matrix.  Added after the first run.
"""

from __future__ import annotations

import argparse
import hashlib
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
from .embed_requests import request_text

LM = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_TOKENS = 256
DIMENSIONS = 128


def unit_rows(x):
    """Rows scaled to unit length (a zero row stays zero)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(norm == 0, 1.0, norm)).astype(np.float32)


def tfidf_matrix(texts, dimensions=DIMENSIONS):
    """TF-IDF of words and bigrams, reduced by a seeded truncated SVD."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    sparse = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=50000,
                             dtype=np.float64).fit_transform(texts)
    k = min(dimensions, sparse.shape[0] - 1, sparse.shape[1] - 1)
    if k < 1:
        return unit_rows(np.zeros((sparse.shape[0], 1)))
    reduced = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=7, random_state=0).fit_transform(sparse)
    return unit_rows(reduced)


def reduce_pca(x, dimensions=DIMENSIONS):
    """Seeded PCA to at most ``dimensions`` components, scaled to unit length."""
    from sklearn.decomposition import PCA

    k = min(dimensions, x.shape[0] - 1, x.shape[1])
    if k < 1:
        return unit_rows(np.zeros((x.shape[0], 1)))
    return unit_rows(PCA(n_components=k, svd_solver="randomized", random_state=0).fit_transform(x.astype(np.float64)))


class MeanHiddenState:
    """Mean of the last hidden states of a causal language model over the non-padding tokens."""

    def __init__(self, model=LM, max_tokens=MAX_TOKENS, device=None, dtype="float16"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(0)
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(model, local_files_only=True, dtype=getattr(torch, dtype))
        self.model.to(self.device).eval()
        self.max_tokens = max_tokens
        self.name = model

    def revision(self):
        """The cache revision the weights were read from (the hash only, never a path)."""
        from huggingface_hub.constants import HF_HUB_CACHE
        ref = Path(HF_HUB_CACHE) / f"models--{self.name.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip() if ref.is_file() else "not recorded"

    def encode(self, texts, batch=16):
        torch = self.torch
        lengths = [len(self.tokenizer(t, truncation=True, max_length=self.max_tokens)["input_ids"]) for t in texts]
        order = np.argsort(lengths, kind="stable")
        out = np.empty((len(texts), self.model.config.hidden_size), np.float32)
        for start in range(0, len(texts), batch):
            idx = order[start:start + batch]
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_tokens, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out[idx] = mean.float().cpu().numpy()
        return out


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(kind, rows_dir, corpus, out, cids=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = MeanHiddenState() if kind == "lm" else None
    if kind == "lm":
        (out / "unreduced").mkdir(exist_ok=True)
        info = dict(kind=kind, model=encoder.name, revision=encoder.revision(), max_tokens=encoder.max_tokens,
                    pooling="mean of the last hidden states over non-padding tokens", device=encoder.device,
                    dtype=encoder.dtype,
                    reduction=f"PCA to at most {DIMENSIONS} components per benchmark, seed 0, then unit length",
                    text="state as JSON, then the instructions of the questions", benchmarks={})
    elif kind == "tfidf":
        info = dict(kind=kind, model="TF-IDF of words and bigrams (sublinear tf, at most 50,000 terms)",
                    reduction=f"truncated SVD to at most {DIMENSIONS} components per benchmark, seed 0, then unit length",
                    text="state as JSON, then the instructions of the questions", benchmarks={})
    else:
        raise ValueError("kind must be tfidf or lm")
    for b in list_benchmarks(corpus):
        cid = b["catalog_id"]
        if cids and cid not in cids:
            continue
        rows = corpus_rows(corpus, rows_dir, cid)
        if len(rows) != int(b["requests"]):
            raise ValueError(f"benchmark {cid}: rows do not match the corpus")
        texts = [request_text(r) for r in rows]
        if kind == "tfidf":
            x = tfidf_matrix(texts)
        else:
            raw_path = out / "unreduced" / f"catalog-{cid:03d}.npy"
            if raw_path.is_file():
                raw = np.load(raw_path, allow_pickle=False)
            else:
                raw = encoder.encode(texts)
                np.save(raw_path, raw)
            if not np.isfinite(raw).all():
                raise ValueError(f"benchmark {cid}: non-finite hidden states")
            x = reduce_pca(raw)
        path = out / f"catalog-{cid:03d}.npy"
        np.save(path, x)
        info["benchmarks"][str(cid)] = dict(requests=int(x.shape[0]), dimensions=int(x.shape[1]),
                                            sha256=checksum(path))
        print(f"  benchmark {cid:3d}: {x.shape}", flush=True)
    (out / "features.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Further request representations for the router")
    ap.add_argument("--kind", required=True, choices=("tfidf", "lm"))
    for name in ("rows", "corpus", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--cids", type=int, nargs="*")
    a = ap.parse_args(argv)
    run(a.kind, a.rows, a.corpus, a.out, set(a.cids or []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
