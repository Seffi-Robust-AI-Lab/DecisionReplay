# Research and experiment plan

For approval. Nothing in this plan has been run. Numbers quoted here are stored results of the earlier rounds or
come from `DATA_AUDIT.md`; each is marked with its file.

## 1. What the paper claims

1. **Resource.** A frozen corpus of per-request probability vectors, token counts, cost and latency for 22
   commercial decision models on 36 NLP benchmarks, with a deterministic simulator that replays ensemble, routing
   and cascade policies from it at no inference cost. How much of the corpus the paper may call "released" follows
   the rights decision (`DATA_AUDIT.md`, section 9).
2. **Diagnostics.** How accurate, calibrated, costly and alike the 22 models are on these tasks.
3. **A first benchmark.** Ten standard methods compared on accuracy, log loss, cost and latency.

The paper proposes no new method. The claim that can fail is the first one: it holds only if the simulator
reproduces the stored results of the earlier rounds (E7). If an anchor does not reproduce, work stops there.

Framing stays on NLP tasks and their evaluation: results are reported per benchmark and per task area, with each
benchmark's own metric beside the common ones.

## 2. Protocol

| Item | Choice |
|---|---|
| Unit of action | The request. One call returns all fields of a request, so a router or cascade decides per request |
| Unit of scoring | The judged field |
| Unit of resampling | The split group, within benchmark and test fold |
| Folds | The five frozen grouped folds. Test fold f, validation fold (f+1) mod 5, three fit folds; predictions stitched over the five rotations |
| Fitting | A method with nothing to tune fits on the four non-test folds, as the stored rows do. A method with a setting (panel size, threshold, k, cost weight) fits on three folds and chooses the setting on the validation fold |
| Averaging | Equal weight per benchmark, then the mean |
| Suites | All 36; 35 without RAGTruth. Sensitivity rows: without ToolRet and BRIGHT (recall only), without ACOS (40 % of judged fields), without the four saturated benchmarks |
| Decision rule | Single models: the provider's reported choice. Combinations: first maximum of the normalised vector. Yes/no: "yes" at 0.5 or above. Both rules are reported for single models |
| Normalisation | As in the stored results: divide by the ordinary sum, then by `math.fsum` |
| Log-loss floor | The stored rule, with the floors 1e-6 and 1e-3 beside it as the stored table has them |
| Primary metric | Log loss, nats per judged field |
| Other metrics | Accuracy; native score from the pinned kit; Brier score; expected calibration error (10 bins); calls, cost in USD per 1,000 requests and latency per request |
| Cost | Sum of the recorded cost of the model–request pairs a policy calls. Calls made only to fit a policy are counted separately |
| Latency | Sum over stages of the slowest call in the stage: one stage gives the parallel figure, one model per stage the serial one |
| Intervals and tests | Paired group bootstrap, 1,999 draws, seed `[20261119, catalog_id]`; percentile interval; centred two-sided test; Holm over the contrasts of section 5 |
| Determinism | Seeds from (base seed, catalogue id, fold, draw); ties to the lower index; one BLAS thread; no timestamps in result files |

## 3. Experiments

| # | Question | Output |
|---|---|---|
| E1 | What is in the corpus, and how does it differ from earlier routing and ensemble corpora? | Corpus statistics table; comparison table. The comparison table needs the top-up reading |
| E2 | How accurate, calibrated and costly is each model? | Accuracy, native score, log loss, Brier, calibration error and reliability curves, cost, latency, per model and task area; saturation flags |
| E3 | How alike are the models? | Pairwise error correlation, Q-statistic, disagreement and double-fault rates, Jensen–Shannon distance; grouped by organisation and by confirmed base model; share of fields where all, a majority, one or no model is right |
| E4 | How do the ten baselines compare? | Main table with intervals against the best single model and against the uniform pool |
| E5 | What do cost and latency buy? | Accuracy–cost and accuracy–latency frontiers from panel size, cascade thresholds and the router's cost weight; cost to match the best single model |
| E6 | Ablations | Panel size 1 to 22 (ranked, greedy, random); label budgets of 0, 4, 12 and all groups; leave-one-area-out transfer; calibration before or after pooling; the sensitivity suites |
| E7 | Is the simulator faithful? | Reproduction of stored numbers at full precision (section 7) |

E3 is the tie to the conference theme on homogenization. It is descriptive: base models are confirmed for 7 of
22 models, so "same base model" covers few pairs.

## 4. Baselines and their settings

| # | Baseline | Fit | Setting chosen on validation |
|---|---|---|---|
| 1 | Best single model per benchmark, recalibrated | four non-test folds | none |
| 2 | Oracle selector; random selector | none | none |
| 3 | Uniform pool of all 22: mean of probabilities; majority vote | none | none |
| 4 | Skill-weighted pool: one reliability per model by EM on agreement | unlabeled outputs of four folds | none |
| 5 | Uniform pool with one temperature per benchmark | fit folds | none |
| 6 | Simplex stacking on recalibrated members, with and without a final temperature | fit folds | none |
| 7 | Greedy forward selection; tree-guided variant from the stored paths | fit folds | panel size, one-standard-error rule |
| 8 | kNN router: the model with the best neighbour score minus a cost weight | fit folds | k; the cost weight traces the frontier |
| 9 | Confidence cascade: models in order of cost, stop when the request's confidence passes a threshold | fit folds | thresholds trace the frontier; at most three stages |
| 10 | Certified early exit on the weights of baseline 6: stop when the uncalled weight cannot change any answer | as baseline 6 | none |

- Baseline 8 has two feature sets: numeric request features (token count, field count, option counts), and a
  sentence embedding of the request text. The second needs local text and cannot be rerun from released files.
- Baseline 9 defines the confidence of a request as the lowest top probability among its fields, after one
  temperature per model. The mean is reported in the appendix.
- Baseline 10 makes the same decisions as the full weighted panel by construction; it is compared on calls, cost
  and latency only.
- Appendix rows, already implemented: confidence-weighted and median pools, ranked top-k, best pair,
  conditional-logit combiner, provider-level vote, random panels by size.

## 5. Named contrasts

Log loss, mean over 36 benchmarks, A − B; negative means A is better. These nine contrasts form the family for
the Holm adjustment. "Known" means that the earlier rounds already estimated it on the same test folds.

| # | A − B | Status | Stored value |
|---|---|---|---|
| C1 | Uniform mean pool − best single | point estimates known | 0.4786 and 0.3491 (`ensemble_round2/results/rows.csv`) |
| C2 | Pool with temperature − uniform pool | known | −0.0703, interval −0.0741 to −0.0668 (`comparisons.csv`, H3b) |
| C3 | Skill-weighted pool − uniform pool | known | −0.0228, interval −0.0231 to −0.0225 (`comparisons.csv`, H6) |
| C4 | Simplex stacking − pool with temperature | point estimates known | 0.3281 and 0.4083 (`rows.csv`) |
| C5 | Simplex stacking with final temperature − best single | point estimates known | 0.3220 and 0.3491 (`rows.csv`) |
| C6 | Tree-guided selection − best single | known | −0.0166, interval −0.0187 to −0.0146 (`statistics/primary_contrasts.csv`) |
| C7 | Tree-guided − greedy selection | known | −0.0002, interval −0.0008 to 0.0004 (same file) |
| C8 | kNN router − best single, at the best single model's cost | new | |
| C9 | Confidence cascade − best single, at the best single model's cost | new | |

Descriptive, outside the family: oracle against best single (the audit gives 96.7 % against 87.9 % of judged
fields); certified early exit against the full panel (calls, cost and latency saved); every contrast on accuracy
and on the native score.

Seven of nine contrasts are known in whole or in part. The paper therefore reports them as a benchmark on a
corpus that has been used before, and says so. Only C8 and C9, and the frontiers of E5, are new results.

## 6. Frozen before any evaluation

One settings file lists every baseline's grid, every seed, the suites and the contrasts above. Its hash is
written into every result file. A change after the first evaluation gets a new hash and a line in the paper's
appendix. The simulator is written and tested on made-up data before it sees the real corpus.

## 7. Reproduction anchors (E7)

Compared at full precision with the stored files, before any new number is trusted:

- the 792 single-model rows and the 36 uniform-pool rows of `expanded-paid-166/baselines.csv` (this audit already
  reproduces the 792 accuracies to 1e-12 with the reported choice);
- the rows of `ensemble_round2/results/rows.csv` and the 576 rows of `anchors.csv`;
- the contrasts C2, C3, C6 and C7 with their intervals;
- cost of the full panel, $3.5102 per 1,000 requests, and the total of $156.975725;
- latency of all 22 in parallel (median 1.269 s, 95th percentile 5.131 s) and in series (10.457 s, 19.489 s).

Tolerance 1e-12 for replay and 1e-6 for refits.

## 8. Limits known now

- **The test folds are not fresh.** Earlier studies evaluated many configurations on them.
- **Release.** 70 of 792 cells are cleared for open release. Results on the other cells can be reported only
  as far as the providers' terms allow; one provider's terms restrict published comparisons.
- **Text stays local.** The embedding router and the native scores cannot be rerun from released files without
  rebuilding the benchmark rows through the pinned kit.
- **Cost and latency are a snapshot** of one collection in November 2026, through one gateway. For two models
  cost depends on the serving provider or on a discount that the corpus does not record.
- **Possible training overlap.** Some decision models may have been trained on these benchmarks. The corpus
  cannot show this; one small model is the most accurate on 8 of 36 benchmarks while ranking 20th on average.
- **No refusals, no generated text, no repeated samples.** Methods that need them are out of scope
  (`DATA_AUDIT.md`, section 11).

## 9. From experiments to the paper

| Paper element | From |
|---|---|
| Table 1: comparison with earlier corpora | E1 and the top-up reading |
| Table 2: corpus statistics by task area | E1 |
| Figure 1: model similarity | E3 |
| Figure 2: calibration and accuracy by model | E2 |
| Table 3: the ten baselines | E4 |
| Figure 3: accuracy–cost and accuracy–latency frontiers | E5 |
| Figure 4: panel size and label budget | E6 |
| Appendix: per-benchmark tables, sensitivity suites, rights matrix, data statement | E2, E4, E6 |

Every number, table and figure is written by a script from result files; the text holds no typed number.

## 10. Decisions this plan needs

1. The ten baselines, and whether the conditional-logit combiner joins them.
2. The headline suite: 36 or 35 benchmarks.
3. Framing (resource paper or evaluation study) and real or coded model names.
