"""What the per-request oracle's lead over the best model per benchmark consists of.

    python3 -B -m analysis.headroom_anatomy --corpus DIR --settings FILE --results RESULTS --out OUT \
        [--processes N]

The paper reports that an oracle which picks the best model for each request is far ahead of the best model per
benchmark (``oracle`` against ``best_single``).  Reviewers asked how much of this lead a policy could ever reach:
an oracle over many models is credited whenever one of them is right, by knowledge or by luck, and a wrong gold
answer puts a question among those that only a few models "get right".  This module takes the lead apart.  It
needs the local corpus and the per-field archives ``RESULTS/local/fields-<method>.npz``; it reads ``RESULTS`` and
writes ``OUT/headroom_anatomy.json`` only.  The design below was fixed before the first run on the corpus.

Every share is an equal-benchmark mean in which a judged field weighs 1 / its benchmark's number of judged fields,
as in ``hardness``.  Per suite the file holds:

``lead``        the oracle's accuracy, the reference's accuracy and their difference (the paper's headroom);
``by_options``  for fields with 2, 3 to 5, and 6 or more options: the share of fields, the accuracy of the
                reference and of the oracle on them, the part of the lead that lies on them, and the two
                question-level ceilings below;
``ceilings``    the share of fields on which at least one, two or three models are right.  "At least two" asks
                for a second witness, so a single lucky model does not count;
``guessing``    of the fields that the reference gets wrong, the share on which at least one other model is right
                (observed), beside the share that M - 1 answerers would reach by guessing uniformly among the
                options, 1 - (1 - accepted / options) ** (M - 1).  If the observed share is no higher than the
                guessing share, the lead by itself is no evidence that the models know different things;
``confident``   the same oracle when a model's answer counts only if the model's own calibrated probability of
                its answer is at least tau (0.5, 0.7, 0.9, 0.99), else the reference's answer stands.  A right
                answer given with a calibrated probability of 0.9 is not a lucky guess.  Calibration is the
                per-model transform of the rotation, fitted on the three fit folds, as for every calibrated
                policy; the probability is the fourth return of ``core.decode``;
``most_confident``  the accuracy of answering every field with the model that is most confident on it (first
                maximum).  This uses no label, so it is a method and not a ceiling; it calls every model;
``few_right``   on the fields that one to three models get right: how often the most confident model is one of
                them, beside the mean share of models that are right there (what a random pick would reach) and
                the reference's accuracy; and the mean calibrated confidence of the right and of the wrong models;
``consensus``   the fields on which at least M - 3 models choose one and the same option that is not accepted:
                their share, and the part of the lead that lies on them.  On such a field either the gold answer
                is doubtful or one distractor is dominant; the oracle's credit there comes from at most three
                dissenting models.  No gold answer is inspected here.  On a question with two options this is
                the same event as "at most three models are right", so the block is also given by number of
                options (``by_options``, added on the same day after the first run, when a check of the note
                pointed this out); the figures for three or more options are the informative ones.

Checks that stop the run: the archives reproduce ``rows.json`` (as in ``hardness``); decoding the recalibrated
vectors of a model gives the hits of its original vectors except on exact ties, which are counted and reported.
Only aggregates are written.  Added after the first run, on 23 November 2026, in answer to a review.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from data.schema import N_FOLDS, sha256_file
from sim import core
from sim.baselines.random_single import request_vectors
from sim.engine import load_task
from . import run_experiments as RX
from .hardness import FEW_RIGHT, load_fields, model_hits

REFERENCE, ORACLE = "best_single", "oracle"
OPTION_GROUPS = (("2", 2, 2), ("3 to 5", 3, 5), ("6 or more", 6, 10 ** 9))
WITNESSES = (1, 2, 3)
TAUS = (0.5, 0.7, 0.9, 0.99)
TOLERANCE = 1e-12


def calibrated_decisions(task):
    """([M, judged] confidence, [M, judged] decoded option, ties) of every model's recalibrated vectors."""
    b = task.b
    choices = np.array([task.recal(f)[0] for f in range(N_FOLDS)])
    confidence = np.empty((b.M, task.nJ))
    option = np.empty((b.M, task.nJ), np.int64)
    hits = np.empty((b.M, task.nJ), bool)
    for m in range(b.M):
        pooled = request_vectors(task, np.full(task.R, m, np.int32), choices)
        hit, _, choice, conf = core.decode(b, pooled)
        confidence[m], option[m], hits[m] = conf[task.J], choice[task.J], hit[task.J]
    return confidence, option, hits


def wrong_consensus(task, option, hits):
    """[judged] the largest number of models that choose one and the same option that is not accepted."""
    b = task.b
    out = np.zeros(task.nJ, np.int64)
    width = task.width
    for w in np.unique(width):
        sel = np.flatnonzero(width == w)
        counts = np.zeros((len(sel), int(w)), np.int64)
        rows = np.arange(len(sel))
        for m in range(b.M):
            np.add.at(counts, (rows, option[m, sel]), (~hits[m, sel]).astype(np.int64))
        out[sel] = counts.max(axis=1)
    return out


def one_benchmark(cid, corpus, results, stored):
    """Sums over the judged fields of one benchmark; every entry is later divided by the number of fields."""
    task = load_task(corpus, cid)
    M, n = task.b.M, task.nJ
    raw_hits = model_hits(task)
    fields = load_fields(results, (REFERENCE, ORACLE), cid, n)
    for name in (REFERENCE, ORACLE):
        if name not in fields:
            raise ValueError(f"no archived fields of {name} for benchmark {cid}")
        if abs(stored[name, cid] - fields[name].mean()) > TOLERANCE:
            raise ValueError(f"{name}: archived hits do not reproduce rows.json on benchmark {cid}")
    ref, oracle = fields[REFERENCE], fields[ORACLE]
    confidence, option, hits = calibrated_decisions(task)
    ties = int((hits != raw_hits).sum())
    right = hits.sum(axis=0)
    width, accepted = task.width, task.n_acc
    part = dict(n=n, M=M, ties=ties, ref=float(ref.sum()), oracle=float(oracle.sum()))

    part["witness"] = {k: float((right >= k).sum()) for k in WITNESSES}
    groups = {}
    for label, low, high in OPTION_GROUPS:
        g = (width >= low) & (width <= high)
        groups[label] = dict(fields=float(g.sum()), ref=float(ref[g].sum()), oracle=float(oracle[g].sum()),
                             **{f"witness_{k}": float((right[g] >= k).sum()) for k in WITNESSES})
    part["groups"] = groups

    # fields the reference gets wrong: is another model right, and what would uniform guessing by M - 1 reach
    missed = ~ref
    guess = 1.0 - (1.0 - accepted / width) ** (M - 1)
    recovered = missed & (right >= 1)
    part["guessing"] = dict(missed=float(missed.sum()), recovered=float(recovered.sum()),
                            guessing=float(guess[missed].sum()))
    part["guessing_groups"] = {}
    for label, low, high in OPTION_GROUPS:
        g = missed & (width >= low) & (width <= high)
        part["guessing_groups"][label] = dict(missed=float(g.sum()), recovered=float((g & (right >= 1)).sum()),
                                              guessing=float(guess[g].sum()))

    part["confident"] = {str(t): float((ref | (hits & (confidence >= t)).any(axis=0)).sum()) for t in TAUS}
    best = confidence.argmax(axis=0)
    by_confidence = hits[best, np.arange(n)]
    part["most_confident"] = float(by_confidence.sum())

    few = (right >= 1) & (right <= FEW_RIGHT)
    part["few"] = dict(
        fields=float(few.sum()), most_confident=float(by_confidence[few].sum()), ref=float(ref[few].sum()),
        random_pick=float((right[few] / M).sum()),
        confidence_right=float(confidence[:, few][hits[:, few]].sum()), pairs_right=float(hits[:, few].sum()),
        confidence_wrong=float(confidence[:, few][~hits[:, few]].sum()), pairs_wrong=float((~hits[:, few]).sum()))

    agreed = wrong_consensus(task, option, hits) >= M - FEW_RIGHT
    part["consensus"] = dict(fields=float(agreed.sum()), oracle=float(oracle[agreed].sum()),
                             ref=float(ref[agreed].sum()), few=float((agreed & few).sum()),
                             none_right=float((agreed & (right == 0)).sum()))
    part["consensus_groups"] = {}
    for label, low, high in OPTION_GROUPS:
        g = (width >= low) & (width <= high)
        part["consensus_groups"][label] = dict(
            fields=float((agreed & g).sum()), group=float(g.sum()), oracle=float(oracle[agreed & g].sum()),
            ref=float(ref[agreed & g].sum()), few=float((agreed & few & g).sum()), few_all=float((few & g).sum()))
    return cid, part


def aggregate(parts, cids):
    """Equal-benchmark summaries: every benchmark contributes its per-field means."""
    B = len(cids)

    def mean(get):
        return math.fsum(get(parts[c]) / parts[c]["n"] for c in cids) / B

    def ratio(top, bottom):
        b = mean(bottom)
        return mean(top) / b if b else None

    M = parts[cids[0]]["M"]
    ref, oracle = mean(lambda p: p["ref"]), mean(lambda p: p["oracle"])
    lead = oracle - ref
    out = dict(benchmarks=B, models=M, reference=REFERENCE,
               lead=dict(oracle=oracle, reference=ref, points=100 * lead),
               ties_changing_a_hit=int(sum(parts[c]["ties"] for c in cids)))
    out["ceilings"] = {f"at_least_{k}": dict(share=mean(lambda p, k=k: p["witness"][k]),
                                             points_over_reference=100 * (mean(lambda p, k=k: p["witness"][k]) - ref))
                       for k in WITNESSES}
    by_options = {}
    for label, _, _ in OPTION_GROUPS:
        def g(key, label=label):
            return lambda p: p["groups"][label][key]
        share = mean(g("fields"))
        by_options[label] = dict(
            share_of_fields=share,
            benchmarks_with_fields=sum(parts[c]["groups"][label]["fields"] > 0 for c in cids),
            reference_accuracy=ratio(g("ref"), g("fields")), oracle_accuracy=ratio(g("oracle"), g("fields")),
            lead_points=100 * (mean(g("oracle")) - mean(g("ref"))),
            share_of_lead=(mean(g("oracle")) - mean(g("ref"))) / lead if lead else None,
            **{f"at_least_{k}": ratio(g(f"witness_{k}"), g("fields")) for k in WITNESSES})
    out["by_options"] = by_options

    def guessing(block):
        missed = mean(lambda p: block(p)["missed"])
        return dict(reference_wrong_share=missed,
                    another_model_right=mean(lambda p: block(p)["recovered"]) / missed if missed else None,
                    uniform_guessing=mean(lambda p: block(p)["guessing"]) / missed if missed else None)

    out["guessing"] = dict(guessing(lambda p: p["guessing"]),
                           by_options={label: guessing(lambda p, label=label: p["guessing_groups"][label])
                                       for label, _, _ in OPTION_GROUPS})
    out["confident"] = {t: dict(accuracy=mean(lambda p, t=t: p["confident"][t]),
                                points_over_reference=100 * (mean(lambda p, t=t: p["confident"][t]) - ref),
                                share_of_question_level_lead=(
                                    (mean(lambda p, t=t: p["confident"][t]) - ref)
                                    / (mean(lambda p: p["witness"][1]) - ref)))
                        for t in map(str, TAUS)}
    most = mean(lambda p: p["most_confident"])
    out["most_confident"] = dict(accuracy=most, points_over_reference=100 * (most - ref))
    few = lambda key: (lambda p: p["few"][key])                                              # noqa: E731
    out["few_right"] = dict(
        share_of_fields=mean(few("fields")), most_confident_right=ratio(few("most_confident"), few("fields")),
        random_pick_right=ratio(few("random_pick"), few("fields")), reference_right=ratio(few("ref"), few("fields")),
        mean_confidence_of_right_models=ratio(few("confidence_right"), few("pairs_right")),
        mean_confidence_of_wrong_models=ratio(few("confidence_wrong"), few("pairs_wrong")))
    con = lambda key: (lambda p: p["consensus"][key])                                        # noqa: E731
    on_agreed = mean(con("oracle")) - mean(con("ref"))
    out["consensus"] = dict(
        models_agreeing_at_least=M - FEW_RIGHT, share_of_fields=mean(con("fields")),
        lead_points=100 * on_agreed, share_of_lead=on_agreed / lead if lead else None,
        share_of_few_right_fields=ratio(con("few"), few("fields")),
        share_of_none_right_fields=ratio(con("none_right"), lambda p: p["n"] - p["witness"][1]))
    out["consensus"]["by_options"] = {}
    for label, _, _ in OPTION_GROUPS:
        def c(key, label=label):
            return lambda p: p["consensus_groups"][label][key]
        on_group = mean(c("oracle")) - mean(c("ref"))
        out["consensus"]["by_options"][label] = dict(
            share_of_group_fields=ratio(c("fields"), c("group")), lead_points=100 * on_group,
            share_of_lead=on_group / lead if lead else None,
            share_of_few_right_fields=ratio(c("few"), c("few_all")))
    return out


def run(corpus, settings_path, results, out, processes=1):
    corpus, results, out = Path(corpus), Path(results), Path(out)
    digest = sha256_file(settings_path)
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    document = json.loads((results / "rows.json").read_text(encoding="utf-8"))
    if document["settings_sha256"] != digest:
        raise SystemExit("rows.json was made with other settings; run the experiments first")
    rows = [r for r in document["rows"] if r["catalog_id"] is not None and r["variant"] is None]
    stored = {(r["method"], r["catalog_id"]): r["accuracy"] for r in rows}
    cids = sorted({r["catalog_id"] for r in rows})
    work = partial(one_benchmark, corpus=corpus, results=results, stored=stored)
    if processes > 1:
        with ProcessPoolExecutor(processes) as pool:
            parts = dict(pool.map(work, cids))
    else:
        parts = dict(map(work, cids))
    suites = {}
    for suite, definition in settings["suites"].items():
        members = [c for c in cids if c not in set(definition["exclude"])]
        if members:
            suites[suite] = aggregate(parts, members)
    RX._json(out / "headroom_anatomy.json", dict(
        synthetic=document["synthetic"], settings_sha256=digest, reference=REFERENCE, oracle=ORACLE,
        option_groups=[label for label, _, _ in OPTION_GROUPS], taus=list(TAUS), few_right=FEW_RIGHT, suites=suites))
    return suites


def main(argv=None):
    ap = argparse.ArgumentParser(description="What the oracle's lead over the best model per benchmark consists of")
    for name in ("corpus", "settings", "results", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args(argv)
    suites = run(a.corpus, a.settings, a.results, a.out, a.processes)
    headline = json.loads(Path(a.settings).read_text(encoding="utf-8"))["headline_suite"]
    block = suites[headline]
    print(f"{block['benchmarks']} benchmarks; lead {block['lead']['points']:.2f} points; "
          f"at least two right {block['ceilings']['at_least_2']['points_over_reference']:.2f}; "
          f"another model right on {100 * block['guessing']['another_model_right']:.1f}% of the reference's errors, "
          f"uniform guessing {100 * block['guessing']['uniform_guessing']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
