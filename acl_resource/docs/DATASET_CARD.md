# Dataset card

## Summary

The corpus supports offline evaluation of policies that select or combine model outputs on NLP benchmarks.
Each prediction is a probability vector over the options of one field in one request. Benchmark text stays
with the benchmark holder. Release `{{release_version}}` covers `{{benchmark_count}}` benchmarks and
`{{model_count}}` models.

## Supported NLP tasks

`{{task_coverage}}` describes the NLP tasks represented in this release. The schema supports choice and
yes/no questions. A request can contain several fields, each with its own options. A policy can be evaluated
on the judged fields after the holder rebuilds the labels.

## Structure and schemas

`benchmarks.csv` identifies each benchmark by catalogue id and records its name, task area, native metric,
request count and field count. The local tier adds the judged-field count. `models.csv` maps each model index
to a stable code, a name, an organisation (the party that offers the model, which the paper calls its provider),
a base model and an input price per token. The names mode is
recorded in `manifest.json`, alongside the schema version, tier, coverage, counts, tables and source binding.
`SHA256SUMS` lists payload file hashes, and the manifest records the hash of that list.

| Parquet table | One row per | Contents |
|---|---|---|
| `items` | Request | Request index, opaque request and group identifiers, fold, field count and row hash |
| `fields` | Field | Request and field indices, question type and option count |
| `splits` | Request | Request index, frozen fold, integer split group and collection run rank |
| `labels` | Field | Request and field indices, judged flag and ordered list of accepted option positions |
| `predictions` | Field and model | Request and field indices, question type, raw probability vector and reported choice |
| `usage` | Request and model | Request index, call count, input and output token counts and cost in USD; in the local and gated tiers also the serving host (column `provider`), the transport and the attempt count |
| `latency` | Request and model | Request index and latency in seconds |

The first four tables sit in `catalog-NNN`. The other three sit in its `model-MM` directories. Exact column
names and types are defined in `acl_resource/data/schema.py`.

## Tiers

| Tier | Model and benchmark coverage | Tables |
|---|---|---|
| Local | All source cells | All seven tables, including locally derived labels |
| Gated | All source cells | Items, fields, splits, predictions and usage |
| Open | Models released on at least one benchmark by the release matrix, crossed with the benchmarks released for all of them | Schema version 2 (the release): items, fields, splits, predictions, usage without the serving host, the transport and the attempt count, and latency. Schema version 1: items, fields, splits and predictions |

Open-tier model indices follow source order, and their codes retain the original numbering. Its manifest
binds the clearance file by hash. The release matrix `data/meta/release_clearance.csv` keeps the review's status
of every cell and adds the authors' release decision (`data/meta/release_decision.json`); `data/meta/release_notices.csv`
holds, for each released model, the review's findings and conditions, which travel with the files. In coded names mode, the model name is its code, and organisation and
base-model codes preserve shared membership.

## How the corpus was collected

`{{collection_protocol}}` describes collection of the source outputs. The builder copies items and predictions
from the checksummed export. It binds local rows to the exported items, derives fields, labels and splits,
and converts the recorded observations into usage and latency for the tiers that carry them.

## How to rebuild labels

Obtain the matching benchmark rows under each benchmark's terms and keep them locally. The row directory
holds one `NNN.jsonl.gz` per included catalogue id. Use a gated or open corpus and choose an empty or new
directory for the private copy.

```sh
python3 -B -m acl_resource.data.rehydrate --corpus CORPUS --rows ROWS --out PRIVATE_COPY
```

The command checks the row binding and compares reconstructed fields and splits with the stored tables.
It writes accepted option positions for judged fields and an empty list for each unjudged field.
The copy retains its original release tier and records `"labels": "rehydrated"`. A copy of the open tier of schema
version 2 then holds what every policy of the simulator needs: vectors, labels, cost and latency.

A release can leave out requests for every model; its manifest then declares their number per benchmark under
`excluded_requests`, and the remaining requests of the benchmark are numbered in their order. The benchmark rows
still hold every request. The command above binds them to the corpus by request id, drops the rows of the requests
that the corpus does not hold, and stops unless their number is the declared one.
Keep this copy on the machine that holds the rows.

## Splits

Each split group stays within one of five frozen folds. In each rotation, one fold is the test fold, the
next fold cyclically is the validation fold and the remaining three are fit folds. The leave-one-area-out
track uses other task areas as sources and keeps benchmarks with specified shared sources in the same area.

## Known limitations

An option position has meaning within its field. Comparisons across requests must account for differences
in the option sets. The schema permits multiple accepted positions and unjudged fields. Raw vectors can
contain zeros and need normalisation for policy replay. A reported choice and the first maximum can differ
on a tie. Release-specific coverage and limitations are `{{known_limitations}}`.

Row binding checks request and native group identifiers, field counts and request order. A revision confined
to accepted answers can pass those checks and produce different labels. Rebuilding the original labels
therefore depends on obtaining the matching revision of the benchmark rows.

## Licensing

Access and reuse depend on the asset and its source. The layers and questions for review are described in
`LICENSE_LAYERS.md`. The applicable release terms and procedure for obtaining benchmark rows are
`{{release_terms}}`.

## Maintenance

`{{maintenance_process}}` describes how to report errors, track revisions and obtain updated manifests.

## Placeholders

| Placeholder | Information to supply for the release |
|---|---|
| `{{release_version}}` | Release identifier |
| `{{benchmark_count}}` | Number of included benchmarks |
| `{{model_count}}` | Number of included models |
| `{{task_coverage}}` | NLP tasks covered by the included benchmarks |
| `{{collection_protocol}}` | Collection procedure and provenance of the export |
| `{{known_limitations}}` | Limitations specific to the included models, benchmarks and collection |
| `{{release_terms}}` | Reviewed terms and benchmark access procedure |
| `{{maintenance_process}}` | Correction, versioning and maintenance process |
