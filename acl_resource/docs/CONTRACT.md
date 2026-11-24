# Package contract

Shared by everyone who writes code for this package. It fixes vocabulary, the frozen files, the data layout,
the programming interfaces and the definitions of every method and metric. If something here is wrong or
missing, do not work around it silently: write it down in your `NOTES.md`.

## 1. What the package is

A corpus and an offline simulator for NLP evaluation. The corpus holds, for every request of 36 NLP benchmarks
(tool use, retrieval, natural-language inference, question answering, classification and others) and for each of
22 models, a probability vector over the answer options of every question in the request, plus token counts,
cost and latency of the call. The simulator replays ensemble, routing and cascade policies from these files. No
model is ever called. Request text and gold answers are not in the corpus; the labels table is derived locally.

Vocabulary used in code, comments and documents:

| Term | Meaning |
|---|---|
| benchmark | one NLP benchmark, identified by its catalogue id (`cid`) |
| task area | one of five groups of benchmarks: knowledge, language, retrieval, tools, arts |
| request | one call to a model: a state and one or more typed questions |
| field | one question of a request; a model returns one probability vector per field |
| option | one possible answer of a field; fields have 2 or more options |
| yes/no field | question type `noul`; its vector is `[1 - p, p]` |
| judged field | a field with at least one accepted option; only judged fields are scored |
| split group | linked requests that must stay in one fold; the unit of resampling |
| policy | a rule that decides which models to call for a request and how to combine their vectors |
| provider | the party that offers a model under its name. The paper's word since 24 November 2026; in the corpus, in result files and in code it is the model's `organisation`, and those names stay |
| host | the party that served a call (the column `provider` of the usage records, which the release does not hold) |

Keep this vocabulary. Do not rename things to abstract decision-theory terms.

## 2. Hard rules

1. Never fabricate a metric, a citation or a telemetry field. A number is either computed by code from files,
   or it is not written. A column that the corpus does not have is not invented.
2. Double-blind anonymity: no person's name, institution, e-mail address, user name, host name or absolute path
   in code, comments, documents, file metadata or test data.
3. Framing stays on NLP tasks and their evaluation.
4. No item text, option text, gold answer or per-item value in any document, log or result file meant for
   release. Result files under `results/*/local/` may hold per-field arrays and never leave the machine.
5. No network access, no new dependency. Available: Python 3.13, numpy, scipy, pandas, pyarrow, scikit-learn,
   matplotlib, pytest.
6. Determinism: seeds derive from (base seed, catalogue id, fold, draw); ties go to the lower index; sums that
   are compared across runs use `math.fsum`; no timestamps, host names or absolute paths in output files; JSON
   is written with sorted keys. Running a command twice gives byte-identical output files.
7. Run everything with `python3 -B` from the folder that contains `acl_resource/`.

## 3. Frozen files (read, import, never edit)

| File | Content |
|---|---|
| `acl_resource/data/schema.py` | table schemas, tiers, paths, the one table writer, `public_id`, `roles` |
| `acl_resource/data/tables.py` | derivation of the processed tables from the source export and the benchmark rows |
| `acl_resource/data/load.py` | `load_benchmark(corpus, cid)` -> `BenchmarkData`; normalisation |
| `acl_resource/data/reference_build.py` | minimal reference build of a tier "local" corpus |
| `acl_resource/sim/types.py` | `BenchmarkData`, `Outcome`, constants, result row keys |
| `acl_resource/sim/core.py` | transforms, decoding, accepted mass, Brier, calibration error, cost of a call plan |
| `acl_resource/synthetic/make_synthetic.py` | the made-up corpus |
| `acl_resource/tests/conftest.py` | test fixtures |
| `acl_resource/SETTINGS.json` | every setting of every experiment |

## 4. Data

Source export (read only): `models.csv`, `benchmarks.csv`, `SHA256SUMS`, `manifest.json`, and per benchmark
`catalog-NNN/items.parquet`, `catalog-NNN/model-MM/predictions.parquet`, `.../observations.parquet`.
Benchmark rows (local only): `NNN.jsonl.gz`. Processed corpus: see the docstring of `schema.py`.

Facts about the real corpus that code must handle (all present in the made-up corpus):

- 22 models, 36 benchmarks; requests have 1 to 150 fields; fields have 2 to 151 options; in half of the
  benchmarks the options differ from item to item, so there are no global classes.
- Yes/no fields exist in two benchmarks. `reported_choice` is null on yes/no fields.
- Some fields are not judged (in two benchmarks most are not, and all judged answers there are the positive
  option). Some judged fields have more than one accepted option.
- Raw vectors need not sum to 1 (off by up to 0.0035), and can hold exact zeros, also on the accepted option.
- A provider's reported choice can differ from the first maximum of its vector (exact ties).
- Usage: one row per request and model. A few requests were answered in several calls (`transport`
  `capacity_adapted`, `provider` null, `calls` > 1). Output tokens are 0 for many models.

Stored-result rules that must be reproduced exactly:

- Normalisation: `load.normalize` (choice: divide by the ordinary sum, then by the exactly rounded sum; yes/no:
  by the exactly rounded sum only).
- Decoding: renormalise by the exactly rounded sum, first maximum; yes/no "yes" when p(yes) >= 0.5.
- Accepted mass: sum of the probabilities of the accepted options; a yes/no field whose answer is "no" takes
  1 - p(yes). Log loss: -log of the mass clipped to [1e-15, 1], mean over judged fields.
- A single model "as collected" is scored on its raw vectors with its reported choice; a single model "as a
  panel of one" is scored on normalised vectors with the first maximum. Both exist in stored results.
- Benchmarks are averaged with equal weight: `math.fsum(values) / n`.

Folds: five frozen folds per benchmark. For test fold f: validation fold (f + 1) mod 5, the other three are the
fit folds (`schema.roles`). Every field is predicted by the rotation that holds its fold out ("stitched").

## 5. Interfaces

### 5.1 `acl_resource/sim/engine.py`

`Task(bench)`: one benchmark with judged-field tables and the recalibration menu. It is a port of the reference
class `ensemble_round2.protocol.Task` onto `BenchmarkData`, with the same attribute and method names and the
same results to the last bit: `b, cid, name, area, F, R, field_fold, J, nJ, jpos, fold, width, n_acc, row_group,
G, group_fold, field_group, group`, and `block_mass, mass_menu, mass, cal_stats, recal_choice, recal,
choice_by_fold, view, rows, fit_loss, stitched_loss, stitch, pool_fields, fold_blocks, mass_of_blocks, decode`.
`row_group` is `bench.split_group`. The menu is `types.MENU`; reuse `core` for transforms, mass and decoding.
Also: `load_task(corpus, cid)` and `load_tasks(corpus, cids=None)`.

### 5.2 `acl_resource/sim/metrics.py`

- `Scored`: `cid, mass [J], hit [J], brier [J], confidence [J], cost (dict from core.plan_costs), info, ledger`.
- `score(task, outcome) -> Scored`: decode `outcome.pooled` with `core.decode`, restrict to judged fields.
- `summarize(scored) -> dict` with the metric keys of `types.RESULT_ROW_KEYS` for one benchmark: log loss at the
  three floors of `types.REPORT_FLOORS`, `n_clipped` (mass below 1e-15), accuracy, mean Brier, calibration
  error (`core.ece` on judged fields), cost and latency. `native` is left `None`: it is filled in locally.
- `suite_rows(rows_by_cid, suites)`: equal-benchmark means for every suite of the settings.

### 5.3 `acl_resource/sim/stats.py`

Ports of the reference statistics: `group_multiplicities`, `interval_and_test`, `holm_adjust`, `select_size`
(reference `jdi/paper_stats.py`) and `bench_replicates`, `suite_contrast`, `paired` (reference
`ensemble_round2/contrasts.py`). Paired group bootstrap: split groups resampled within benchmark and test fold,
1,999 draws, seed `[20261119, cid]`; percentile interval; centred two-sided test; Holm over the contrast family.

### 5.4 `acl_resource/sim/baselines/`

One module per method. Each exposes `NAME` and `run(task, settings) -> Outcome` (all five rotations, stitched),
and `acl_resource/sim/baselines/__init__.py` exposes `BASELINES = {name: module}`. A method that needs other
benchmarks exposes `run_suite(tasks, settings) -> {cid: Outcome}` instead. `Outcome.info` records what was
chosen in each rotation (model, weights, size, threshold, temperature) as JSON-ready values; `Outcome.ledger`
starts from `types.empty_ledger()` and says what the method saw of the target benchmark.

### 5.5 Command lines

```
python3 -B -m acl_resource.data.build_corpus --source DIR --rows DIR --meta DIR --out DIR --tier local|gated|open [--clearance FILE] [--names real|coded]
python3 -B -m acl_resource.data.build_corpus --verify DIR
python3 -B -m acl_resource.analysis.release --review DIR --decision FILE --out acl_resource/data/meta [--check]
python3 -B -m acl_resource.analysis.build_handoff --out DIR --results DIR --corpus DIR --open DIR --pdf real=FILE [--zip FILE]
python3 -B -m acl_resource.analysis.diagnostics --corpus DIR --out DIR
python3 -B -m acl_resource.analysis.run_experiments --corpus DIR --settings acl_resource/SETTINGS.json --out DIR [--cids 1 2 ...] [--only rows|contrasts|frontier|ablations] [--processes N]
python3 -B -m acl_resource.analysis.export_paper_assets --corpus DIR --results DIR --diagnostics DIR --json name=FILE [--json ...] --related-work DIR --out acl_resource/paper/generated [--check]
```

## 6. Methods

"fit" = the three fit folds, "validation" = the validation fold, "train4" = the four non-test folds, all of the
rotation at hand. "calibrated" = each member's vectors pass through the menu transform chosen for it by
`Task.recal` (cross-fitted inside the folds it was fitted on, deployment transform elsewhere). Reference names
are functions of the reference code that the port must match bit for bit where one is given.

| Name | Definition | Reference |
|---|---|---|
| `best_single` | The model with the lowest mean judged log loss on train4, calibrated with the four-fold trace; its calibrated vectors on the test fold | `anchors.ranking(T, f, "calibrated", n=4)`; stored row `context_single_calibrated_four_folds` |
| `oracle` | Per request, the model with the most correct judged fields (ties: lower index); a bound that reads test labels | new |
| `random_single` | Per request, one model drawn uniformly (seeded) | new |
| `uniform_mean` | Equal-weight mean of the 22 original vectors | stored row `context_flat_mean_22` |
| `majority_vote` | Each model votes for its decoded option; the pooled vector is the vote share | new |
| `skill_pool` | One reliability per model by EM on the votes of train4 (labels unread), turned into pool weights; linear pool of original vectors | `reliability.em_skills`, `pool_weights`, `fit_benchmark` key `main`; stored row `e6_skill_pool` |
| `uniform_temperature` | `uniform_mean`, then one temperature per rotation from the 33-value menu, fitted on the fit folds | `calibration.flat_tables`, `fits.best_temperature`, as `temperature_3b_by_benchmark` in `calibration.main`; stored row `e3b_flat_mean_temperature_fit_folds` |
| `stack` | Simplex weights maximising the log score of the pool of calibrated members on the fit folds; variant `stack_final_temperature` adds one temperature chosen on cross-fitted pool predictions | `calibration.stack_control(cid, "simplex")`, `fits.simplex_fit`; stored rows `e3_stack_recalibrated`, `e3_stack_recalibrated_final_temperature` |
| `greedy` | Forward selection with replacement on the calibrated mass of the fit folds (22 prefixes); the prefix size is chosen on the validation fold by the one-standard-error rule | `anchors.caruana_path`, `paper_stats.select_size` |
| `knn_router` | One model per request from its nearest fit requests; see below | new |
| `confidence_cascade` | Models in order of cost, stop when the request's confidence passes a threshold; see below | new |
| `certified_exit` | The `stack` panel called one model at a time in order of decreasing weight; stop when the uncalled weight cannot change any decoded answer of the request | reference excerpt `certified_depths` |

Call plans: a linear pool calls every model with positive weight in stage 0. `best_single`, `oracle`,
`random_single` and `knn_router` call one model in stage 0. `confidence_cascade` and `certified_exit` call one
model per stage.

**`greedy`, size selection.** Utilities are minus the mean calibrated log loss of each prefix on the validation
fold. 499 bootstrap resamples of the validation fold's split groups, from
`numpy.random.default_rng([110000 + 100 * cid + f])`, multinomial over the groups; `select_size` picks the size.

**`knn_router`.** Request features: `settings["features"]` is `"numeric"` (log1p of the median input tokens
over models, field count, mean and maximum number of options, share of yes/no fields; standardised with the mean
and standard deviation of the fit requests) or a folder with `catalog-NNN.npy` ([R, d] float32, local only).
The training pool is the fit requests that have a judged field. Utility of model m on request n: minus the mean
calibrated log loss of m on the judged fields of n. Score of m for a test request: mean utility over its k
nearest pool requests (Euclidean distance, ties to the lower request index) minus lambda times the relative cost
of m (its mean cost per request on the fit folds divided by the largest such mean). The request goes to the
model with the highest score (ties: lower index); the answer is that model's calibrated vectors. k is chosen on
the validation fold at lambda = 0 from the grid of the settings, by mean log loss. For the main table lambda is
the value of the grid with the lowest validation log loss among those whose validation cost does not exceed
the validation cost of that rotation's `best_single` model (the largest lambda if none qualifies).

**`confidence_cascade`.** Per rotation: compute for every model its mean cost per request and its mean
calibrated judged log loss on the fit folds; take the models that are not dominated in both; order them by
cost; the chain is the cheapest, the one in the middle and the one with the lowest loss (fewer if the set is
smaller). Confidence of a model on a request: the lowest top probability among the request's fields, on its
calibrated vectors. Call the first model; stop if its confidence is at least tau, else call the next; the last
model always answers. The answer is the calibrated vectors of the model that stopped the chain. One tau for all
stages, from the grid of the settings (0 = always stop at the first model, a value above 1 = always reach the
last). For the main table tau is chosen on the validation fold like lambda above.

**`certified_exit`.** Members are the models with positive `stack` weight in the rotation (no final
temperature), in order of decreasing weight (ties: lower index). After each call, a field is certified when the
lead of its best option over the second best, in the partial weighted sum of calibrated vectors, exceeds the
total weight not yet called (with the floating-point guard of the reference); a yes/no field is certified when
the partial sum cannot cross 0.5 either way. Stop when every field of the request is certified. The answer is
the partial weighted pool. Decoded answers must equal those of the full `stack` panel on every field; a test
asserts it.

## 7. Experiments (driven by `SETTINGS.json`)

- **Rows.** Every method of `settings["methods"]` on every benchmark, then suite means for every suite.
- **Contrasts.** Each entry of `settings["contrasts"]`: A minus B on the suite, for log loss and accuracy, with
  the paired group bootstrap; Holm over the log-loss family.
- **Frontier.** Points (calls, cost, latency, accuracy, log loss) for: ranked top-k equal-weight pools of
  calibrated members, k = 1..22 (ranking by calibrated log loss on the fit folds); `greedy` prefixes k = 1..22;
  random panels of each size (seeds of the settings, averaged); `knn_router` over the lambda grid;
  `single_by_cost` over the same grid (the router with every fit request as a neighbour, which is one
  cost-penalised model per benchmark; a control added after the first run); `confidence_cascade` over the tau
  grid.
- **Ablations.** (a) Order of calibration for the equal-weight pool: none; final temperature; calibrated
  members; calibrated members and final temperature. (b) Label budget: `best_single` and `stack` fitted on n
  split groups sampled from train4, n from the settings, averaged over the settings' draws, and on all of
  train4. (c) No target labels: `transfer_single` (the model with the lowest equal-benchmark mean of
  whole-benchmark original log loss over the benchmarks of the other four task areas) and
  `transfer_temperature` (`uniform_mean` with the menu temperature that minimises the equal-benchmark mean loss
  on the other four areas).

- **Operating points** (`analysis/operating_points.py`, added after the first run). For `knn_router`, its text
  variant, `single_by_cost`, `confidence_cascade` and the cascade with mean confidence: the main-table rule
  ("capped") and the grid value with the lowest validation loss at any cost ("uncapped"), per rotation; how often
  each grid value was chosen, how often no value was within the budget and how often the cap changed the choice;
  paired contrasts against `best_single` and against the control (descriptive, not adjusted). Output:
  `operating_points.json`. The two baselines return a selection record per rotation in `Outcome.info` and accept a
  list with one parameter per fold; `settings["confidence_cascade"]["confidence"]` is `"lowest"` (default) or
  `"mean"`.

- **Redrawn folds** (`analysis/fold_redraws.py`, added after the first run). The source groups of every
  benchmark are shuffled (`numpy.random.default_rng([draw, cid])`) and dealt into five folds; every built-in
  method is refitted on each draw. Output: `fold_redraws.json` with each method's suite row per draw beside its
  row on the frozen folds, and the contrasts between built-in methods per draw.
- **Open track** (`analysis/open_track.py`, added after the first run). On an open-tier corpus with labels
  rebuilt by `data.rehydrate`: `best_single`, `oracle`, `uniform_mean`, `majority_vote`, `skill_pool`,
  `uniform_temperature`, `stack`, `stack_final_temperature` and `greedy`, log loss and accuracy only (the open tier has
  no usage). Output: `open_track.json`.
- **Release matrix** (`analysis/release.py`, added after the first run). From the review of terms (read only) and
  the authors' decision `data/meta/release_decision.json` it writes `data/meta/release_clearance.csv` (the review's
  cell matrix unchanged, plus `release` = released or held and `release_basis`), `data/meta/release_notices.csv` (the
  review's findings and conditions for each released model) and `data/meta/release_summary.json` (the counts the
  paper quotes). A cell is released when its model is not held and the review found no source clause that bars the
  anonymous positional results; the released cells must form a rectangle. `build_corpus --tier open` selects the
  cells whose `release` is `released`; a clearance file without that column selects `numeric_public_status` =
  `cleared`. `reproduce.sh` checks with `--check` that the matrix still follows from the review and the decision.
- **Homogenization, additions** (`analysis/homogenization.py`): the correlation of two models' errors inside
  strata of judged fields on which the same number of the other models are wrong, pooled over strata; and, on
  the judged fields with at least two wrong options that no model answers correctly, the share on which all
  models, and at least three quarters of them, choose the same wrong option.
- **Released corpus and the paper** (decisions of the authors, 23 November 2026). The paper reports the released
  corpus only: the 17 released models on the 35 released benchmarks, under the listed model names, in one build.
  - *Open tier, schema version 2* (`data/schema.py`: `PUBLIC_SCHEMA_VERSION`, `PUBLIC_USAGE`, `PUBLIC_TABLES`,
    `corpus_tables`, `usage_schema`; `build_corpus --tier open --names real --schema-version 2`): items, fields,
    splits, predictions and, for every request and released model, `usage` (calls, input and output tokens, billed
    cost) and `latency`. No serving host, transport or attempt count. `lint_release` and `build_corpus --verify`
    refuse a version-2 usage table with other columns. Version 1 (vectors only) still builds.
  - *Requests removed for every model* (decision of 24 November 2026). `data/meta/release_decision.json` lists
    `exclude_host_requests`: a released model, a serving host and the number of requests that the host served for
    it. `build_corpus --schema-version 2 --decision FILE` finds these requests in the source export (the serving
    host of every request and model), stops unless their number is the expected one, leaves them out for every
    model and numbers the remaining requests of a benchmark in their order; splits are derived from the remaining
    requests. The manifest declares the number per benchmark (`excluded_requests`). Benchmark rows hold every
    request, so they are bound to such a corpus by request id, never by position: `data.tables.row_positions` and
    `rows_of`, used by `data.rehydrate` (labels from the full rows), `data.load.corpus_rows`, the own-metric scorer
    (`sim/native.py`, through `BenchmarkData.item_id`), the leakage count and the text featurizers. The request
    representations in `data/features` have one row per request of the rows; `analysis/features_for_corpus.py`
    writes those of the corpus's requests to `data/features/release` (matrices of untouched benchmarks copied,
    per-request encoders subset, TF-IDF and PCA fitted again on the kept requests), which
    `SETTINGS_RELEASE.json` names for the text router. `analysis/release.py` checks the number against the
    review's serving records and adds the removal to the notices and to the summary (`requests_released`,
    `excluded_requests*`, `secondary_host_*_in_corpus`).
  - *Pipeline* `reproduce_release.sh RESULTS [PROCESSES]`: release check, open tier of version 2
    (`RESULTS/corpus_open`), labels rebuilt from the benchmark rows (`RESULTS/corpus`), then every step of
    `reproduce.sh` except the open track and the comparison with stored results. Settings: `SETTINGS_RELEASE.json`,
    which is `SETTINGS.json` with the panel sizes of 17 models and `replay_stored_fits: false` (the three methods
    with stored fits were fitted on 22 models and are not replayed; the planned contrasts C6 and C7 fall away, and
    the Holm family has seven members). On the released corpus `reproduce_followup.sh` takes three settings from the
    environment: `SETTINGS=acl_resource/SETTINGS_RELEASE.json`, `FEATURES=acl_resource/data/features/release` (the
    request representations of the corpus's requests) and `FULL_TEXT_ALL=acl_resource/results/followup/local/features`
    (the whole-request representation of every request, from which the corpus's is derived).
    Results: `results/release/main`, `results/release/followup`.
  - *Audit* `analysis/audit_release.py` writes `docs/AUDIT_RELEASE.json` in the layout of `docs/AUDIT_SUMMARY.json`,
    recomputed from the corpus tables; `--check-against-audit` runs the same code on the local tier of the whole
    collection and must reproduce the stored audit. With `--local` it adds what the open tier does not carry (the
    number of attempts, and the count of requests billed off the listed price that the model's usual host served).
  - *Paper build* `build_paper.sh RESULTS CORPUS` reads `SETTINGS_RELEASE.json` and `docs/AUDIT_RELEASE.json` by
    default (`SETTINGS`, `AUDIT` override). The paper prints two references, Best Single (key `T`) and the best
    model per benchmark (key `B`, method `best_single`); the oracles and the gains against them are still computed
    and are not printed. `results/main`, `results/followup`, `SETTINGS.json`, `docs/AUDIT_SUMMARY.json` and
    `reproduce.sh` are the record of the first study on 22 models and 36 benchmarks; the paper build does not read
    them.
- **Figures 1 and 2** (`analysis/overview_figures.py`). Since 24 November 2026 the overview and the modules strip
  are drawn by code in the design of the hand-drawn figures of the first study and written by the exporter as TikZ
  (`paper/generated/figures/overview.tex`, `pipeline.tex`): the totals from the audit, the grid from the corpus's
  lists of models and benchmarks, the dots from the per-model accuracy, cost and median latency of the headline
  suite. The export stops when these inputs disagree, when a dot falls outside its strip or when a text does not
  fit; the manifest records the inputs of both pictures. The hand-drawn figures show the 22 models of the first
  study and are kept, with their checker `analysis/static_figures.py`, in `paper/static_figures_22_models/`, which
  no package takes. A drawing placed in `paper/static_figures/` would still be checked against the data on every
  export, but the paper includes none.

Result files in the output folder: `rows.json`, `contrasts.json`, `frontier.json`, `ablations.json`,
`run.json` (hash of the settings file, hash of the corpus manifest, whether the corpus is made-up), and
`local/fields-<method>.npz` with per-field mass and hit arrays (never released). Each JSON file is
`{"synthetic": bool, "settings_sha256": str, "rows": [...]}` with the row keys of `types.py`.

## 8. Diagnostics (output folder of `analysis.diagnostics`)

Aggregates only, never a per-item value.

| File | One row per | Columns (at least) |
|---|---|---|
| `models.csv` | benchmark and model | accuracy by reported choice and by first maximum; log loss on raw and on normalised vectors, at the three floors; Brier; calibration error; mean confidence; share of zero entries; judged fields with zero mass on the accepted options; usd per 1,000 requests; latency p50 and p95; calls and attempts per request |
| `reliability.csv` | benchmark, model and confidence bin | count, mean confidence, accuracy |
| `pairs.csv` | benchmark and model pair | error correlation, Q-statistic, disagreement rate, double-fault rate, mean Jensen-Shannon distance, same organisation, same base model |
| `headroom.csv` | benchmark | share of judged fields where: the best single model in hindsight, at least one, at least two, a majority, all, no model is right; mean number of models right; effective number of voters M / (1 + (M - 1) * mean pairwise error correlation) |
| `failures.csv` | benchmark and model | shares of judged fields: right; wrong with confidence >= 0.9; wrong with confidence < 0.9; zero mass on the accepted options; tied maxima |
| `diagnostics.json` | | suite means of all of the above by model, by task area and overall; `{"synthetic": bool}` |

Pairwise statistics use the first-maximum rule on judged fields. A statistic that is undefined (a model that is
always right, an empty cell) is written as an empty value, never as 0.

## 9. Tests

`python3 -B -m pytest acl_resource/tests -q -p no:cacheprovider` must pass. Tests use the made-up corpus only.
`conftest.py` gives the fixtures `test_root` (folder with the layout of the made-up corpus), `corpus`
(`test_root/processed/local`) and `planted` (the planted structure). If the environment variable
`ACL_TEST_ROOT` is set, that folder is used; otherwise the corpus is generated into a temporary folder.
Parity tests against reference code import the reference packages from `ACL_TEST_ROOT` and skip when they are
not there. They call only pure functions of the reference, never a `main`.

### 7.x Additions of the restyle (21 November 2026, after the first run)

- `analysis/hardness.py` -> `hardness.json`: per suite, judged fields by the number of models right (0..M), each
  method's accuracy per count (every field weighs 1 / its benchmark's judged fields), the fields with one to three
  models right, and the oracle restricted to k models (top-k by the ranking of `ranked_top_k`; 200 seeded random
  subsets with the 5th and 95th percentiles). Checks: archived hits reproduce `rows.json`; no single-model method is
  right where no model is; the oracle over all models reproduces the stored oracle.
- `analysis/router_study.py` -> `router_study.json`: `router_<representation>` and `router_control` rows under the
  capped and uncapped rules; recall at 1, 3, 5 of a right model among the router's top-ranked models; routing shares.
  Representations are read from `acl_resource/data/features/<name>/` (local; built by `embed_requests` and
  `featurize_requests`).
- `analysis/benchmark_metrics.py` -> `benchmark_metrics.json`: references R, T, B, D, O; Gain@R/T/B, Gap@O with
  paired group-bootstrap intervals for Gain@B; PerfGain and CostSave against T and B (best settings chosen on the
  test folds, as in LLMRouterBench); ParetoDist on min-max normalised axes, cost on a log scale.
  Keys and paper symbols differ: key `T` (`gain_t`, `perf_gain_T`, `cost_save_T`) is the paper's Best Single
  $\mathcal{B}$, one model for all benchmarks chosen on the test folds; key `B` (`gain_b`, `perf_gain_B`,
  `cost_save_B`) and the method `best_single` are the paper's $\mathcal{B}_d$, the best model per benchmark chosen
  on the non-test folds; `D` is the benchmark oracle $\mathcal{D}^{\star}$, `O` the per-request oracle
  $\mathcal{O}$, `R` the random model $\mathcal{R}$. The keys are kept as they are so that earlier result files stay
  comparable.
