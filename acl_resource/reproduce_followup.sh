#!/bin/sh
# Rebuild the follow-up analyses of 23 November 2026 from the frozen inputs and the main results.  Run from the
# folder that contains acl_resource/, after reproduce.sh.
#
#   sh acl_resource/reproduce_followup.sh RESULTS_DIR FOLLOWUP_DIR [PROCESSES]
#
# Local inputs: the corpus and the request representations of reproduce.sh, and the benchmark rows (for the
# benchmarks' own metrics and for the whole-request representation).  The whole-request representation runs the
# local sentence encoder over 412,250 text windows; set FULL_TEXT to a folder that already holds it to skip that.
# For a corpus from which requests were removed (the released corpus): FEATURES names the representations of its
# requests (reproduce_release.sh writes acl_resource/data/features/release), and FULL_TEXT_ALL names a folder that
# holds the whole-request representation of every request; the one of the corpus's requests is derived from it.
set -eu
RESULTS="$1"
OUT="$2"
PROCESSES="${3:-4}"
CORPUS="$RESULTS/corpus"
[ -d "$CORPUS" ] || CORPUS=acl_resource/data/processed/local
SETTINGS="${SETTINGS:-acl_resource/SETTINGS.json}"       # SETTINGS_RELEASE.json for the released pool
ROWS=data/decision_index/common-paid-65/inputs
FEATURES="${FEATURES:-acl_resource/data/features}"
FULL_TEXT="${FULL_TEXT:-$OUT/local/features/minilm_full}"
mkdir -p "$OUT"
if [ -n "${FULL_TEXT_ALL:-}" ] && [ ! -f "$FULL_TEXT/features.json" ]; then
  python3 -B -m acl_resource.analysis.features_for_corpus --features "$FULL_TEXT_ALL" --rows "$ROWS" --corpus "$CORPUS" --out "$(dirname "$FULL_TEXT")"
fi
cp "$RESULTS/rows.json" "$OUT/rows.json"
python3 -B -m acl_resource.analysis.fold_redraws --corpus "$CORPUS" --settings "$SETTINGS" --out "$OUT" \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.headroom_anatomy --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.recall_references --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.pool_composition --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.router_refit_features --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" \
  --features "$FEATURES" --rows "$ROWS" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.router_unknown --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" \
  --features "$FEATURES/minilm" --out "$OUT" --project-root . --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.router_control_contrast --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" \
  --features "$FEATURES/minilm" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.own_metric_study --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" \
  --rows "$ROWS" --project-root . --features "$FEATURES/minilm" --out "$OUT" --processes "$PROCESSES" > /dev/null
if [ ! -f "$FULL_TEXT/features.json" ]; then
  python3 -B -m acl_resource.analysis.embed_requests_full --rows "$ROWS" --corpus "$CORPUS" --out "$FULL_TEXT" --device cpu
fi
python3 -B -m acl_resource.analysis.router_full_text --corpus "$CORPUS" --settings "$SETTINGS" --results "$RESULTS" \
  --old "$FEATURES/minilm" --new "$FULL_TEXT" --out "$OUT" --processes "$PROCESSES" > /dev/null
python3 -B -m acl_resource.analysis.followup_facts --followup "$OUT"
