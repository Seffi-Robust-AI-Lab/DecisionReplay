#!/bin/sh
# Rebuild every result of the released pool: the 17 released models on the 35 released benchmarks, from the open
# tier of schema version 2 (probability vectors, usage and latency) with labels rebuilt from the benchmark rows.
# Run from the folder that contains acl_resource/.
#
#   sh acl_resource/reproduce_release.sh RESULTS_DIR [PROCESSES]
#
# Local inputs as for reproduce.sh.  The methods with stored fits are not part of this run: their weights were
# fitted on the pool of all models (SETTINGS_RELEASE.json, "replay_stored_fits").  The settings are those of
# SETTINGS.json with the panel sizes of 17 models.  reproduce.sh and results/main stay the record of the first study.
#
# The release decision removes some requests for every model (data/meta/release_decision.json,
# "exclude_host_requests").  The corpus builder leaves them out, and every later step binds the benchmark rows to
# the corpus by request id.  The request representations in acl_resource/data/features hold every request of the
# rows; acl_resource/data/features/release gets the same representations for the requests of this corpus (written
# again by every run).
set -eu
OUT="$1"
PROCESSES="${2:-4}"
SOURCE=results/decision_index/paper/dataset/candidate
ROWS=data/decision_index/common-paid-65/inputs
META=acl_resource/data/meta
SETTINGS=acl_resource/SETTINGS_RELEASE.json
RIGHTS=results/decision_index/paper/rights
CLEARANCE=acl_resource/data/meta/release_clearance.csv
FEATURES_ALL=acl_resource/data/features
FEATURES=acl_resource/data/features/release       # named by SETTINGS_RELEASE.json for the text router
OPEN="$OUT/corpus_open"
CORPUS="$OUT/corpus"
START=$(date +%s)

mkdir -p "$OUT"
python3 -B -m acl_resource.analysis.release --review "$RIGHTS" --decision "$META/release_decision.json" --out "$META" --check
python3 -B -m acl_resource.data.build_corpus --source "$SOURCE" --rows "$ROWS" --meta "$META" --out "$OPEN" --tier open --clearance "$CLEARANCE" --names real --schema-version 2 --decision "$META/release_decision.json"
python3 -B -m acl_resource.data.build_corpus --verify "$OPEN"
python3 -B -m acl_resource.data.rehydrate --corpus "$OPEN" --rows "$ROWS" --out "$CORPUS"
python3 -B -m acl_resource.analysis.features_for_corpus --features "$FEATURES_ALL" --rows "$ROWS" --corpus "$CORPUS" --out "$FEATURES"
python3 -B -m acl_resource.data.splits --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT"
python3 -B -m acl_resource.analysis.leakage --rows "$ROWS" --corpus "$CORPUS" --out "$OUT/leakage.json"
python3 -B -m acl_resource.analysis.diagnostics --corpus "$CORPUS" --out "$OUT/diagnostics"
python3 -B -m acl_resource.analysis.homogenization --corpus "$CORPUS" --out "$OUT/diagnostics" --settings "$SETTINGS"
python3 -B -m acl_resource.analysis.run_experiments --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --processes "$PROCESSES"
python3 -B -m acl_resource.analysis.local_methods --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --project-root . --processes "$PROCESSES"
python3 -B -m acl_resource.analysis.run_experiments --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --only contrasts --processes "$PROCESSES"
python3 -B -m acl_resource.analysis.operating_points --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --project-root . --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.fold_redraws --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.hardness --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.router_study --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --features "$FEATURES" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.benchmark_metrics --corpus "$CORPUS" --results "$OUT" --settings "$SETTINGS" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.paper_facts --results "$OUT" --diagnostics "$OUT/diagnostics" --settings "$SETTINGS" --out "$OUT/facts.json" > /dev/null
python3 -B -c "
import json, platform, sys, time
import numpy, scipy, pandas, pyarrow, sklearn
json.dump(dict(seconds=int(time.time()) - $START, processes=$PROCESSES, cpu_count=__import__('os').cpu_count(),
               python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
               pandas=pandas.__version__, pyarrow=pyarrow.__version__, scikit_learn=sklearn.__version__),
          open('$OUT/timing.json', 'w'), indent=1, sort_keys=True)"
echo "done: $OUT"
