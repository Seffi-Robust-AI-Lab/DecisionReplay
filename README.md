# DecisionReplay

A corpus of recorded model outputs and an offline simulator for comparing ensembles, routers and cascades on NLP
benchmarks.

The corpus covers 17 commercial decision models from 6 providers on
35 fixed-choice benchmarks: 106,845 requests with 267,716 questions. A decision model is
an endpoint that returns a probability for every answer option. For every request and model the corpus holds the
probability vector of each question, the number of calls, the token counts, the billed cost and the measured
latency. A policy is replayed on these records, so no model is called again.

## Paper

**DecisionReplay: A Corpus and Offline Simulator for Combining, Routing and Cascading Decision Models**

arXiv: https://arxiv.org/abs/2611.34627

> Comparing ensembles, routers and cascades requires the output of every model on every item, and its cost.
> Stored collections built for routing keep no per-option probabilities. We introduce DecisionReplay, a
> dataset on which ensemble methods, routers and cascades are simulated quickly, without new model calls. It
> holds the probability vectors of 17 commercial decision models, endpoints that return a probability for
> every answer option, on 35 fixed-choice NLP benchmarks, with the token counts, billed cost and measured
> latency of every call, all released. A deterministic simulator replays any policy that selects, combines or
> cascades them, and the main experiments rerun in 812 seconds on a laptop. One well-chosen model per
> benchmark is hard to beat: pooling all models loses 2.7 accuracy points to it, and stacking gains 1.0 points
> at 3.4 times its cost. The models are wrong together: two that are both wrong choose the same wrong option
> in 72% of cases. The best router matches that model at 64% of its cost, and certified early exit cuts the
> cost of a stacked ensemble by 43% without changing an answer, though not its latency. DecisionReplay makes
> it cheap and fast to test whether a new method changes these results.

The results of the paper, its numbers, tables and figures, are computed from the corpus, from its audit
(`docs/AUDIT_RELEASE.json`) and from the files in `results/`. `reproduce_release.sh` and `reproduce_followup.sh`
are the scripts that wrote those files, with the settings in `SETTINGS_RELEASE.json`.

## What is here

| Path | Content |
|---|---|
| `data/processed/open/` | the corpus: `benchmarks.csv`, `models.csv`, `manifest.json` and the Parquet tables of each benchmark and model (`docs/DATASET_CARD.md`) |
| `data/` | schema, loader, corpus builder, `rehydrate.py` (rebuilds the labels), `hub.py` (tables for a dataset hub), and `meta/` with the release decision and its notices |
| `sim/` | the simulator: metrics, statistics and the policies |
| `analysis/` | the experiments and the export of tables and figures |
| `results/` | the aggregate result files of the paper: `main/` for the main run and `followup/` for the follow-up analyses |
| `synthetic/`, `tests/` | the generator of a made-up corpus, and the tests that run on it |
| `docs/` | the data contract, the dataset card, the experiment plan |
| `SETTINGS.json`, `reproduce.sh` | the settings and the script of an earlier run on a larger pool of models; the settings of the paper's run are derived from them |

The comments in the scripts and `docs/CONTRACT.md` use the folder names of the machine that ran the experiments.
There the results of the paper are `results/release/main` and `results/release/followup`, and `results/main` is
the earlier run, which is not here.

## What is not here

Benchmark text, option text and gold answers. The corpus knows a request by an opaque id and an option by its
position. Scoring a policy needs labels, and they are rebuilt on your machine from the benchmark rows (see
"Labels"). Also not here: the outputs of the models and of the one benchmark that were collected and not
released, and the serving host of any call.

## Start

Tested with Python 3.13 and the library versions in `requirements.txt`. Run every command from the root of the
repository.

    python3 -B -m pip install -r requirements.txt
    python3 -B -m data.build_corpus --verify data/processed/open
    python3 -B -m pytest tests -q -p no:cacheprovider

The second command checks every file of the corpus against its checksum list and prints the counts. The tests
need no data: they generate a made-up corpus. Some are skipped, because they compare with inputs that are not
distributed.

Reading one benchmark:

```python
from data import load

corpus = "data/processed/open"
print(load.read_manifest(corpus)["counts"])
b = load.load_benchmark(corpus, 4)       # catalogue id 4; benchmarks.csv lists the ids
print(b.name, b.R, b.F, b.M)             # requests, questions, models
width = b.widths[0]
print(b.P[width].shape)                  # questions with that many options, models, options
print(b.cost.sum(axis=0))                # billed cost of each model in USD
```

`sim/types.py` describes every array of a loaded benchmark.

## Labels

    python3 -B -m data.rehydrate --corpus data/processed/open --rows ROWS --out PRIVATE_COPY

`ROWS` is a folder with one file `NNN.jsonl.gz` per benchmark, named by catalogue id, that holds the rows of the
benchmark. You obtain them under each benchmark's own terms. The command binds the rows to the requests by id,
checks the binding and writes a private copy of the corpus with the accepted option positions. Keep that copy on
the machine that holds the rows.

With the private copy, the experiments run as in `reproduce_release.sh`, from the line that calls `data.splits`.
The lines before it rebuild the corpus from the collection's export, which is not distributed (`INPUTS.json`
lists the local inputs). The router study also reads request representations, which
`analysis/embed_requests.py` and `analysis/featurize_requests.py` build from the rows. The scripts keep them in
`acl_resource/data/features`, the folder that the settings name: the settings files are kept byte for byte,
because every result file records the checksum of its settings.

## Tables for a dataset hub

    python3 -B -m data.hub --corpus data/processed/open --out HUB_FOLDER --card docs/HUB_CARD.md --notices data/meta/release_notices.csv

This writes the same records as one Parquet file per table and benchmark, with the benchmark and the model as
columns, and a dataset card beside them. Every file is compared with the corpus as it is written.
