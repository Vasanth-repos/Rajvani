"""
verify_consistency.py

A pre-render consistency gate for the Rajvani eval report pipeline.

The problem this exists to solve: across five separate benchmark reports in
this project, the same failure shape kept recurring in different places —
two panels on one page disagreeing about the same number (Mewati WER, TTS
MOS "Pending" vs "4.18/5"), a stale panel sitting next to a freshly-computed
one (the cross-dialect transfer matrix diagonal), baseline-vs-fine-tuned
pairs landing on suspicious exact ties (MT BLEU/chrF++), a stated range not
matching its own source table (the leaderboard's "33.2-57.2"), and values
oscillating between report runs with no explanation (MOS: 4.2 -> 4.6 -> 4.2).

Rather than manually re-spotting the Nth instance of this in a screenshot,
this script checks a SINGLE authoritative eval-run JSON artifact for
internal consistency, and against the previous run's artifact for
unexplained drift, BEFORE any markdown/HTML report is generated from it.

## How this is meant to be wired in

Your report generator (wherever it currently reads per-dialect numbers and
renders benchmark/leaderboard.md, the demo app's Evaluation tab, etc.)
should be changed to read from ONE canonical JSON file per run — see the
RunReport schema below — instead of each panel independently formatting its
own copy of the numbers. Then, as a build/CI step:

    python verify_consistency.py --run eval/runs/latest.json \
                                  --history-dir eval/runs/

If this exits non-zero, the report generator should refuse to render and
should print this script's output instead. Wire it as a `pytest` test
(`tests/test_eval_consistency.py` can just shell out to this script and
assert exit code 0) and as a `make check` step per this project's existing
conventions.

## Canonical run schema (what your generator needs to produce)

{
  "run_id": "2026-08-20T14:03:00",
  "commit_hash": "abc1234",
  "dialects": ["MWR", "MTR", "DHD", "HDT", "MWT", "BGR"],
  "per_dialect": {
    "MWR": {
      "samples": 34,
      "baseline_wer": 13.46, "finetuned_wer": 6.63,
      "wer_ci_95": [4.63, 8.62],
      "cer": 4.18,
      "baseline_bleu": 44.2, "finetuned_bleu": 44.2,   # example: matches source of the "identical BLEU" bug
      "baseline_chrf": null, "finetuned_chrf": 44.2,
      "mos": {"score": 4.30, "voice_model": "MMS-TTS-Dialect", "n_raters": 11, "baseline_score": 2.73, "baseline_voice_model": "gTTS-Hindi-Fallback"},
      "provisional": true
    },
    "...": "..."
  },
  "pooled": {
    "wer": 7.55, "wer_ci_95": [6.57, 8.56],
    "cer": 4.58, "bleu": 57.2, "chrf": 70.6
  },
  "cross_dialect_transfer_matrix": {
    "MWR": {"MWR": 6.63, "MTR": 11.2, "DHD": 12.5, "HDT": 14.1, "MWT": 16.8, "BGR": 18.2},
    "...": "..."
  },
  "leaderboard": {
    "mt_bleu_range_ours": [33.2, 57.2]
  },
  "notes": {
    "baseline_change_reason": null
  }
}

Adapt field names to whatever your actual generator already produces if
they differ — the important thing is ONE file, ONE run, everything reading
from it. This script's field names are a suggestion, not a requirement;
edit the accessors below to match your real schema if needed, but do not
skip the checks themselves.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HARD = "HARD"   # blocks render — exit non-zero
SOFT = "SOFT"   # printed as a warning, does not block


class Issue:
    def __init__(self, severity, code, message):
        self.severity = severity
        self.code = code
        self.message = message

    def __str__(self):
        return f"[{self.severity}] {self.code}: {self.message}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_pooled_aggregation(run, issues, tol=0.15):
    """Pooled WER/BLEU/CER/chrF must be a real sample-weighted average of the
    per-dialect rows — this is the exact class of bug that made the earlier
    Overall Macro Average rows untrustworthy until they were fixed."""
    per = run.get("per_dialect", {})
    pooled = run.get("pooled", {})
    if not per or not pooled:
        issues.append(Issue(HARD, "POOLED_MISSING", "Missing per_dialect or pooled section — cannot verify aggregation."))
        return

    total_samples = sum(d.get("samples", 0) for d in per.values())
    if total_samples == 0:
        issues.append(Issue(HARD, "NO_SAMPLES", "Total sample count across all dialects is zero."))
        return

    for metric, pooled_key, weight_key in [
        ("finetuned_wer", "wer", "samples"),
        ("cer", "cer", "samples"),
        ("finetuned_bleu", "bleu", "samples"),
    ]:
        vals = [(d.get(metric), d.get(weight_key, 0)) for d in per.values() if d.get(metric) is not None]
        if not vals or pooled_key not in pooled:
            continue
        weighted_sum = sum(v * w for v, w in vals)
        weight_sum = sum(w for _, w in vals)
        recomputed = weighted_sum / weight_sum if weight_sum else None
        stated = pooled[pooled_key]
        if recomputed is not None and abs(recomputed - stated) > tol:
            issues.append(Issue(
                HARD, "POOLED_MISMATCH",
                f"Pooled '{pooled_key}' is stated as {stated}, but recomputing a sample-weighted "
                f"average from per_dialect gives {recomputed:.2f} (tolerance {tol})."
            ))


def check_baseline_finetuned_not_suspiciously_identical(run, issues):
    """A baseline and fine-tuned value landing on the exact same number
    (MT BLEU 57.2 == 57.2, chrF++ 70.6 == 70.6, both marked '-' for
    improvement) is the signature of an eval that didn't actually re-run,
    or a value copy-pasted into the wrong slot — not a real result."""
    per = run.get("per_dialect", {})
    pooled = run.get("pooled", {})

    # Pooled-level check (this is where the BLEU/chrF tie actually showed up)
    for base_key, fine_key, label in [("bleu", "finetuned_bleu_pooled", "pooled BLEU"), ]:
        pass  # pooled dict in the schema above only stores one bleu value; see per-dialect check below instead

    for dialect, d in per.items():
        for base_key, fine_key, label in [
            ("baseline_bleu", "finetuned_bleu", "BLEU"),
            ("baseline_chrf", "finetuned_chrf", "chrF++"),
        ]:
            b, f = d.get(base_key), d.get(fine_key)
            if b is not None and f is not None and abs(b - f) < 1e-6:
                if not d.get("no_change_expected", False):
                    issues.append(Issue(
                        HARD, "SUSPICIOUS_TIE",
                        f"{dialect}: baseline and fine-tuned {label} are identical ({b}). "
                        f"Verify the fine-tuned eval actually ran for this dialect/metric "
                        f"(set no_change_expected=true on this record if this is a confirmed real tie)."
                    ))


def check_transfer_matrix_diagonal_matches_finetuned(run, issues, tol=0.5):
    """The transfer matrix's diagonal cell (dialect X trained-and-evaluated on
    itself) should equal that dialect's fine-tuned WER in the main breakdown
    table. A stale matrix generated in an earlier run and never refreshed is
    exactly what produces a mismatch here."""
    matrix = run.get("cross_dialect_transfer_matrix", {})
    per = run.get("per_dialect", {})
    for dialect, row in matrix.items():
        diag = row.get(dialect)
        finetuned = per.get(dialect, {}).get("finetuned_wer")
        if diag is None or finetuned is None:
            continue
        if abs(diag - finetuned) > tol:
            issues.append(Issue(
                HARD, "STALE_MATRIX_DIAGONAL",
                f"{dialect}: transfer matrix diagonal shows {diag}% but the per-dialect "
                f"breakdown's fine-tuned WER is {finetuned}% for the same dialect (diff > {tol}pt). "
                f"The matrix was likely not regenerated in this run."
            ))


def check_leaderboard_range_matches_source(run, issues, tol=0.5):
    """A stated 'our BLEU range' on a public leaderboard has to match the
    actual min/max of the per-dialect numbers it's supposedly summarizing."""
    per = run.get("per_dialect", {})
    stated = run.get("leaderboard", {}).get("mt_bleu_range_ours")
    if not stated:
        return
    bleus = [d.get("finetuned_bleu") for d in per.values() if d.get("finetuned_bleu") is not None]
    if not bleus:
        return
    actual_min, actual_max = min(bleus), max(bleus)
    stated_min, stated_max = stated
    if abs(actual_min - stated_min) > tol or abs(actual_max - stated_max) > tol:
        issues.append(Issue(
            HARD, "LEADERBOARD_RANGE_MISMATCH",
            f"Leaderboard states BLEU range {stated_min}-{stated_max}, but the actual per-dialect "
            f"min/max in this run is {actual_min}-{actual_max}."
        ))


def check_ci_plausibility(run, issues):
    """Flags two patterns seen in earlier reports: (a) relative CI half-widths
    suspiciously uniform across all dialects, consistent with a formulaic
    +/-X% overlay rather than a real per-dialect bootstrap, and (b) a pooled
    CI that barely narrows despite pooling many more samples than any single
    dialect, which real statistics would not produce."""
    per = run.get("per_dialect", {})
    rel_half_widths = []
    for dialect, d in per.items():
        wer = d.get("finetuned_wer")
        ci = d.get("wer_ci_95")
        if wer and ci and len(ci) == 2:
            half_width = (ci[1] - ci[0]) / 2
            rel_half_widths.append(half_width / wer)

    if len(rel_half_widths) >= 3:
        cv = (statistics.pstdev(rel_half_widths) / statistics.mean(rel_half_widths)) if statistics.mean(rel_half_widths) else 0
        if cv < 0.05:
            issues.append(Issue(
                SOFT, "CI_TOO_UNIFORM",
                f"Relative CI half-widths across dialects vary by only {cv*100:.1f}% (coefficient of "
                f"variation). Real per-dialect bootstrap intervals computed from independent samples "
                f"usually show more spread than this — verify the CI code is a genuine per-utterance "
                f"bootstrap, not a formulaic +/-X% overlay."
            ))

    pooled = run.get("pooled", {})
    pooled_ci = pooled.get("wer_ci_95")
    pooled_wer = pooled.get("wer")
    if pooled_ci and pooled_wer and rel_half_widths:
        pooled_rel_hw = (pooled_ci[1] - pooled_ci[0]) / 2 / pooled_wer
        avg_dialect_rel_hw = statistics.mean(rel_half_widths)
        if avg_dialect_rel_hw and pooled_rel_hw / avg_dialect_rel_hw > 0.85:
            issues.append(Issue(
                SOFT, "POOLED_CI_NOT_NARROWING",
                f"Pooled WER's relative CI half-width ({pooled_rel_hw*100:.1f}%) is barely narrower than "
                f"the average per-dialect relative half-width ({avg_dialect_rel_hw*100:.1f}%), despite "
                f"pooling ~{sum(d.get('samples',0) for d in per.values())} samples across {len(per)} "
                f"dialects. Genuine pooling of independent samples should narrow the interval "
                f"substantially more than this — verify the pooled CI is computed from the full sample "
                f"set, not copied/approximated from a single dialect's interval."
            ))


def check_mos_scope_labeled(run, issues):
    """Every MOS figure must state which voice model it's rating and how
    many raters produced it — this is the direct fix for the earlier
    'Pending Eval' vs '4.18/5, n=11' contradiction, which existed because
    nothing enforced that MOS numbers carry their own scope."""
    per = run.get("per_dialect", {})
    for dialect, d in per.items():
        mos = d.get("mos")
        if mos is None:
            continue
        for required_field in ("score", "voice_model", "n_raters"):
            if mos.get(required_field) in (None, ""):
                issues.append(Issue(
                    HARD, "MOS_UNSCOPED",
                    f"{dialect}: MOS entry is missing '{required_field}' — every MOS value must state "
                    f"the voice model and rater count it reflects, or it's not distinguishable from a "
                    f"stale/placeholder number."
                ))


def check_against_history(run, history_dir, issues, drift_tol=0.5):
    """Compares this run's baselines and MOS values against the most recent
    prior run. A baseline moving between runs isn't inherently wrong — the
    baseline methodology can legitimately change — but it must be logged,
    or it reads as the reference point shifting to flatter the improvement
    number. A MOS value oscillating between runs with the SAME n_raters and
    SAME voice model, with no logged reason, means the number isn't a real
    persisted measurement."""
    if not history_dir:
        return
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return

    prior_files = sorted(
        [p for p in history_dir.glob("*.json") if p.name != Path(run.get("_source_path", "")).name],
        key=lambda p: p.stat().st_mtime,
    )
    if not prior_files:
        return
    prior = load_json(prior_files[-1])

    reason_logged = bool(run.get("notes", {}).get("baseline_change_reason"))

    cur_per = run.get("per_dialect", {})
    prior_per = prior.get("per_dialect", {})

    for dialect, d in cur_per.items():
        pd = prior_per.get(dialect)
        if not pd:
            continue

        cur_baseline = d.get("baseline_wer")
        prior_baseline = pd.get("baseline_wer")
        if cur_baseline is not None and prior_baseline is not None:
            if abs(cur_baseline - prior_baseline) > drift_tol and not reason_logged:
                issues.append(Issue(
                    HARD, "UNEXPLAINED_BASELINE_DRIFT",
                    f"{dialect}: baseline WER changed from {prior_baseline}% (previous run "
                    f"{prior.get('run_id')}) to {cur_baseline}% (this run) with no "
                    f"notes.baseline_change_reason logged. Log why the baseline changed, or "
                    f"investigate whether this is unintended drift."
                ))

        cur_mos = d.get("mos", {})
        prior_mos = pd.get("mos", {})
        if (cur_mos.get("n_raters") and prior_mos.get("n_raters")
                and cur_mos.get("n_raters") == prior_mos.get("n_raters")
                and cur_mos.get("voice_model") == prior_mos.get("voice_model")
                and cur_mos.get("score") is not None and prior_mos.get("score") is not None
                and abs(cur_mos["score"] - prior_mos["score"]) > 0.05
                and not reason_logged):
            issues.append(Issue(
                HARD, "MOS_OSCILLATION",
                f"{dialect}: MOS changed from {prior_mos['score']} to {cur_mos['score']} between runs "
                f"with the SAME rater count ({cur_mos['n_raters']}) and SAME voice model "
                f"('{cur_mos['voice_model']}'), and no reason logged. A fixed rater panel scoring the "
                f"same voice should not produce a different persisted average without new ratings "
                f"being added — verify this isn't reading from a different/regenerated source file."
            ))


def check_outliers(run, issues):
    """Flags a dialect whose relative WER improvement is a statistical
    outlier vs. the rest — not necessarily wrong, but worth a specific
    manual re-check rather than accepting it because the table looks fine
    overall (this is how the Bagri 67%-vs-~48% gap should have been caught
    automatically instead of spotted by eye)."""
    per = run.get("per_dialect", {})
    reductions = {}
    for dialect, d in per.items():
        b, f = d.get("baseline_wer"), d.get("finetuned_wer")
        if b and f and b > 0:
            reductions[dialect] = (b - f) / b

    if len(reductions) < 4:
        return
    values = list(reductions.values())
    q1, q3 = statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[2]
    iqr = q3 - q1
    for dialect, r in reductions.items():
        if iqr > 0 and (r > q3 + 1.5 * iqr or r < q1 - 1.5 * iqr):
            issues.append(Issue(
                SOFT, "OUTLIER_IMPROVEMENT",
                f"{dialect}: relative WER reduction ({r*100:.1f}%) is a statistical outlier vs. the "
                f"other dialects (IQR {q1*100:.1f}%-{q3*100:.1f}%). Not necessarily wrong — worth a "
                f"specific manual re-check of this dialect's eval run before trusting it."
            ))


def check_provisional_flagging(run, issues, min_samples=50):
    """Any dialect below the project's own n>=50 reliability bar must be
    marked provisional — dropping the flag once numbers look respectable
    (rather than once n actually crosses 50) was flagged as a real issue
    in an earlier report."""
    per = run.get("per_dialect", {})
    for dialect, d in per.items():
        n = d.get("samples", 0)
        if n < min_samples and not d.get("provisional", False):
            issues.append(Issue(
                HARD, "MISSING_PROVISIONAL_FLAG",
                f"{dialect}: n={n} is below the project's n>={min_samples} reliability bar, but this "
                f"record isn't marked provisional=true. Mark it, or grow the dev set."
            ))


# ---------------------------------------------------------------------------

def run_all_checks(run, history_dir):
    issues = []
    check_pooled_aggregation(run, issues)
    check_baseline_finetuned_not_suspiciously_identical(run, issues)
    check_transfer_matrix_diagonal_matches_finetuned(run, issues)
    check_leaderboard_range_matches_source(run, issues)
    check_ci_plausibility(run, issues)
    check_mos_scope_labeled(run, issues)
    check_against_history(run, history_dir, issues)
    check_outliers(run, issues)
    check_provisional_flagging(run, issues)
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Path to the current run's canonical JSON artifact.")
    parser.add_argument("--history-dir", default=None,
                         help="Directory of prior run JSON artifacts, for drift/oscillation checks.")
    args = parser.parse_args()

    run = load_json(args.run)
    run["_source_path"] = args.run

    issues = run_all_checks(run, args.history_dir)

    hard = [i for i in issues if i.severity == HARD]
    soft = [i for i in issues if i.severity == SOFT]

    if not issues:
        print("All consistency checks passed — no HARD or SOFT issues found.")
        sys.exit(0)

    if hard:
        print(f"\n{len(hard)} BLOCKING issue(s) — report generation should NOT proceed:\n")
        for i in hard:
            print(f"  {i}")
    if soft:
        print(f"\n{len(soft)} warning(s) — review before trusting this run:\n")
        for i in soft:
            print(f"  {i}")

    print(f"\n{len(hard)} hard, {len(soft)} soft. ", end="")
    if hard:
        print("Exiting non-zero — do not render/publish this report until these are resolved.")
        sys.exit(1)
    else:
        print("No blocking issues — safe to render, but review the warnings above.")
        sys.exit(0)


if __name__ == "__main__":
    main()
