#!/bin/sh
# Rebuild every result from the frozen inputs.  Run from the root of the repository.
#
#   sh reproduce.sh RESULTS_DIR [PROCESSES]
#
# Local inputs (see INPUTS.json): the source export, the benchmark rows, the stored results of the
# earlier studies, the review of provider and source terms, and the request representations in
# acl_resource/data/features (embed_requests and featurize_requests build them once from the benchmark rows).
# The open tier is selected by data/meta/release_clearance.csv, which analysis.release
# derives from the review and the authors' release decision (data/meta/release_decision.json).
# Two runs into two folders must give identical files, except timing.json.
set -eu
OUT="$1"
PROCESSES="${2:-4}"
SOURCE=results/decision_index/paper/dataset/candidate
ROWS=data/decision_index/common-paid-65/inputs
META=data/meta
SETTINGS=SETTINGS.json
RIGHTS=results/decision_index/paper/rights
CLEARANCE=data/meta/release_clearance.csv
FEATURES=acl_resource/data/features
CORPUS="$OUT/corpus"
OPEN="$OUT/corpus_open"
OPEN_LABELLED="$OUT/corpus_open_labelled"
START=$(date +%s)

mkdir -p "$OUT"
python3 -B -m data.build_corpus --source "$SOURCE" --rows "$ROWS" --meta "$META" --out "$CORPUS" --tier local
python3 -B -m data.build_corpus --verify "$CORPUS"
python3 -B -m data.splits --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT"
python3 -B -m analysis.leakage --rows "$ROWS" --out "$OUT/leakage.json"
python3 -B -m analysis.diagnostics --corpus "$CORPUS" --out "$OUT/diagnostics"
python3 -B -m analysis.homogenization --corpus "$CORPUS" --out "$OUT/diagnostics" --settings "$SETTINGS"
python3 -B -m analysis.run_experiments --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --processes "$PROCESSES"
python3 -B -m analysis.local_methods --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --project-root . --processes "$PROCESSES"
python3 -B -m analysis.run_experiments --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --only contrasts --processes "$PROCESSES"
python3 -B -m analysis.operating_points --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --project-root . --processes "$PROCESSES" > /dev/null
python3 -B -m analysis.fold_redraws --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m analysis.hardness --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m analysis.router_study --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" --features "$FEATURES" --processes "$PROCESSES" > /dev/null
python3 -B -m analysis.benchmark_metrics --corpus "$CORPUS" --results "$OUT" --settings "$SETTINGS" --processes "$PROCESSES" > /dev/null
python3 -B -m analysis.release --review "$RIGHTS" --decision "$META/release_decision.json" --out "$META" --check
python3 -B -m data.build_corpus --source "$SOURCE" --rows "$ROWS" --meta "$META" --out "$OPEN" --tier open --clearance "$CLEARANCE" --names coded
python3 -B -m data.rehydrate --corpus "$OPEN" --rows "$ROWS" --out "$OPEN_LABELLED"
python3 -B -m analysis.open_track --corpus "$OPEN_LABELLED" --settings "$SETTINGS" --out "$OUT/open_track.json" > /dev/null
python3 -B -m analysis.anchors --corpus "$CORPUS" --results "$OUT" --diagnostics "$OUT/diagnostics" --project-root . --out "$OUT/anchors.json" || echo "reproduction check: see $OUT/anchors.json"
python3 -B -m analysis.paper_facts --results "$OUT" --diagnostics "$OUT/diagnostics" --settings "$SETTINGS" --out "$OUT/facts.json" > /dev/null
python3 -B -c "
import json, platform, sys, time
import numpy, scipy, pandas, pyarrow, sklearn
json.dump(dict(seconds=int(time.time()) - $START, processes=$PROCESSES, cpu_count=__import__('os').cpu_count(),
               python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
               pandas=pandas.__version__, pyarrow=pyarrow.__version__, scikit_learn=sklearn.__version__),
          open('$OUT/timing.json', 'w'), indent=1, sort_keys=True)"
echo "done: $OUT"
