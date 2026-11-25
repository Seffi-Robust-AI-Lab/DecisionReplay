---
pretty_name: DecisionReplay
configs:
{{configs}}
---

# DecisionReplay

Recorded outputs of {{model_count}} commercial decision models from {{provider_count}} providers on
{{benchmark_count}} fixed-choice NLP benchmarks. A decision model is an endpoint that returns a probability for
every answer option. For each of {{requests}} requests and each model, the dataset holds the probability vector
of every question of the request, the number of calls, the input and output token counts, the billed cost and the
measured latency. With these records, a policy that selects or combines models (an ensemble, a router, a cascade)
is evaluated offline, with no new model call.

The dataset holds no benchmark text, no option text and no gold answers. A request is known by an opaque id and
an option by its position. Labels are rebuilt on the user's machine from the benchmark rows (see "Labels").

## Tables

{{table_rows}}

A request can hold several questions; the tables call a question a field ({{fields}} in all). The benchmarks
cover these task areas: {{areas}}.

```python
from datasets import load_dataset

predictions = load_dataset(REPOSITORY, "predictions", split="train")   # REPOSITORY: the name of this dataset
models = load_dataset(REPOSITORY, "models", split="train")
```

## Columns

{{table_columns}}

- `catalog_id` is the benchmark and `model_index` the model. The configurations `benchmarks` and `models` give
  their names, the task area and native metric of a benchmark, and the provider (column `organisation`), base
  model and listed input price of a model. `benchmarks.csv` and `models.csv` hold the same two lists as text.
- `item_index` numbers the requests of a benchmark and `field_index` the fields of a request.
- `question_type` is `choice` or `noul`. A `noul` field is a yes/no question; its vector is `[1 - p, p]`.
- `probabilities` is the vector as the model returned it, one entry per option, in the option order of the
  benchmark. It can contain zeros and is not normalised.
- `reported_choice` is the position of the option that the model reported, where it reported one.
- `fold` is the frozen fold of the request (0 to 4). Requests with the same `split_group` are in the same fold.
  `run_rank` is the rank of the request in the collection's run order.
- `item_id`, `split_group_id`, `native_group_id` and `row_sha256` are opaque: benchmark rows are bound to
  requests by them.
- `calls`, `input_tokens`, `output_tokens` and `cost_usd` are the totals of a request for one model, and
  `latency_s` is its measured latency in seconds. The serving host of a call is not recorded here.

## Requests that are left out

`corpus_manifest.json` is the manifest of the corpus from which these tables were written (schema
`{{schema_version}}`). Under `excluded_requests` it declares, per benchmark, the requests that the release leaves
out for every model: {{excluded_requests}} requests in {{excluded_benchmarks}} benchmarks. The remaining requests
of a benchmark are numbered in their order.

## Labels

The code that accompanies the corpus rebuilds the labels. A user obtains the rows of a benchmark under that
benchmark's own terms; `data/rehydrate.py` of the code binds them to the requests by id, checks the binding and
writes the accepted option positions into a private copy. That command and the simulator read the corpus in its
own layout, one file per benchmark and model, which comes with the code. `data/hub.py` wrote the tables here from
that layout and compared every file with it.

## Limits

- An option position has meaning within its field only. Comparisons across requests must account for different
  option sets.
- Raw vectors need normalising before a policy is replayed.
- A reported choice and the first maximum of the vector can differ on a tie.
- Binding the rows checks request and group identifiers, field counts and request order. A revision of a
  benchmark that changes only the accepted answers passes these checks and gives different labels.
- `release_notices.csv` holds, for each model, what a review of the provider's terms found and the conditions it
  names.
