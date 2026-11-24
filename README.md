# DecisionReplay

A corpus of recorded model outputs and an offline simulator for comparing ensembles, routers and cascades on NLP
benchmarks.

The corpus covers 17 commercial decision models from 6 providers on
35 fixed-choice benchmarks: 106,845 requests with 267,716 questions. A decision model is
an endpoint that returns a probability for every answer option. For every request and model the corpus holds the
probability vector of each question, the number of calls, the token counts, the billed cost and the measured
latency. A policy is replayed on these records, so no model is called again.

## What is here

| Path | Content |
|---|---|
| `acl_resource/data/processed/open/` | the corpus: `benchmarks.csv`, `models.csv`, `manifest.json` and the Parquet tables of each benchmark and model (`acl_resource/docs/DATASET_CARD.md`) |
| `acl_resource/data/` | schema, loader, corpus builder, `rehydrate.py` (rebuilds the labels), `hub.py` (tables for a dataset hub), and `meta/` with the release decision and its notices |
| `acl_resource/sim/` | the simulator: metrics, statistics and the policies |
| `acl_resource/analysis/` | the experiments and the export of tables and figures |
| `acl_resource/results/` | the aggregate result files |
| `acl_resource/synthetic/`, `acl_resource/tests/` | the generator of a made-up corpus, and the tests that run on it |
| `acl_resource/docs/` | the data contract, the dataset card, the experiment plan |

## What is not here

Benchmark text, option text and gold answers. The corpus knows a request by an opaque id and an option by its
position. Scoring a policy needs labels, and they are rebuilt on your machine from the benchmark rows (see
"Labels"). Also not here: the outputs of the models and of the one benchmark that were collected and not
released, and the serving host of any call.

## Start

Tested with Python 3.13 and the library versions in `requirements.txt`. Run every command from this folder.

    python3 -B -m pip install -r requirements.txt
    python3 -B -m acl_resource.data.build_corpus --verify acl_resource/data/processed/open
    python3 -B -m pytest acl_resource/tests -q -p no:cacheprovider

The second command checks every file of the corpus against its checksum list and prints the counts. The tests
need no data: they generate a made-up corpus. Some are skipped, because they compare with inputs that are not
distributed.

Reading one benchmark:

```python
from acl_resource.data import load

corpus = "acl_resource/data/processed/open"
print(load.read_manifest(corpus)["counts"])
b = load.load_benchmark(corpus, 4)       # catalogue id 4; benchmarks.csv lists the ids
print(b.name, b.R, b.F, b.M)             # requests, questions, models
width = b.widths[0]
print(b.P[width].shape)                  # questions with that many options, models, options
print(b.cost.sum(axis=0))                # billed cost of each model in USD
```

`acl_resource/sim/types.py` describes every array of a loaded benchmark.

## Labels

    python3 -B -m acl_resource.data.rehydrate --corpus acl_resource/data/processed/open --rows ROWS --out PRIVATE_COPY

`ROWS` is a folder with one file `NNN.jsonl.gz` per benchmark, named by catalogue id, that holds the rows of the
benchmark. You obtain them under each benchmark's own terms. The command binds the rows to the requests by id,
checks the binding and writes a private copy of the corpus with the accepted option positions. Keep that copy on
the machine that holds the rows.

With the private copy, the experiments run as in `acl_resource/reproduce_release.sh`, from the line that calls
`acl_resource.data.splits`. The lines before it rebuild the corpus from the collection's export, which is not
distributed (`acl_resource/INPUTS.json` lists the local inputs). The router study also reads request
representations, which `acl_resource/analysis/embed_requests.py` and `featurize_requests.py` build from the rows.

## Tables for a dataset hub

    python3 -B -m acl_resource.data.hub --corpus acl_resource/data/processed/open --out HUB_FOLDER --card acl_resource/docs/HUB_CARD.md --notices acl_resource/data/meta/release_notices.csv

This writes the same records as one Parquet file per table and benchmark, with the benchmark and the model as
columns, and a dataset card beside them. Every file is compared with the corpus as it is written.

## Status

The paper that describes the corpus is not published yet. No licence has been chosen for the code or for the
corpus. `acl_resource/data/meta/release_notices.csv` holds, for each model, what a review of the provider's terms
found, and `acl_resource/docs/LICENSE_LAYERS.md` lists the questions that a licence has to answer.
