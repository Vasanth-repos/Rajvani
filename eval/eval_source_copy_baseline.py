"""
eval/eval_source_copy_baseline.py

Computes the Source-Copy / No-Translation baseline:
Directly evaluates raw `text_dialect` (untranslated) against `reference_hindi` across the 200 held-out test records.
Quantifies natural dialect-Hindi lexical overlap without any model in the loop.
"""

import json
import sys
from pathlib import Path
import numpy as np
import sacrebleu

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.stdout.reconfigure(encoding="utf-8")

def compute_bootstrap_ci(data: list, num_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 42):
    if not data:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    n = len(data)
    arr = np.array(data)
    boot_means = np.empty(num_bootstrap)
    for i in range(num_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return (round(lower, 2), round(upper, 2))

def run_source_copy_eval(test_path: str = "data/realworld_test_200.jsonl"):
    with open(ROOT_DIR / test_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"=== Running Source-Copy / No-Translation Baseline (Total Utterances: {len(records)}) ===")

    by_dialect = {}
    all_srcs = []
    all_refs = []
    per_utt_bleu = []
    per_utt_chrf = []

    for r in records:
        did = r["dialect"]
        src = r["text_dialect"]
        ref = r.get("reference_hindi", r.get("text_hindi", src))

        s_bleu = sacrebleu.sentence_bleu(src, [ref]).score
        s_chrf = sacrebleu.sentence_chrf(src, [ref]).score

        if did not in by_dialect:
            by_dialect[did] = {
                "srcs": [],
                "refs": [],
                "bleu_scores": [],
                "chrf_scores": []
            }

        by_dialect[did]["srcs"].append(src)
        by_dialect[did]["refs"].append(ref)
        by_dialect[did]["bleu_scores"].append(s_bleu)
        by_dialect[did]["chrf_scores"].append(s_chrf)

        all_srcs.append(src)
        all_refs.append(ref)
        per_utt_bleu.append(s_bleu)
        per_utt_chrf.append(s_chrf)

    print("\n================================================================================")
    print("=== RAW SOURCE-COPY (NO TRANSLATION) OVERLAP BASELINE -> HINDI ===")
    print("================================================================================")

    results = {}
    for did in sorted(by_dialect.keys()):
        data = by_dialect[did]
        c_bleu = sacrebleu.corpus_bleu(data["srcs"], [data["refs"]]).score
        c_chrf = sacrebleu.corpus_chrf(data["srcs"], [data["refs"]]).score
        b_ci_lo, b_ci_hi = compute_bootstrap_ci(data["bleu_scores"])
        c_ci_lo, c_ci_hi = compute_bootstrap_ci(data["chrf_scores"])

        results[did] = {
            "n": len(data["srcs"]),
            "bleu": round(c_bleu, 2),
            "bleu_ci": [b_ci_lo, b_ci_hi],
            "chrf": round(c_chrf, 2),
            "chrf_ci": [c_ci_lo, c_ci_hi]
        }

        print(f"Dialect {did.upper()} (N={len(data['srcs'])}):")
        print(f"  BLEU:   {c_bleu:.2f} (95% CI: [{b_ci_lo:.2f}, {b_ci_hi:.2f}])")
        print(f"  chrF++: {c_chrf:.2f} (95% CI: [{c_ci_lo:.2f}, {c_ci_hi:.2f}])")

    pooled_bleu = sacrebleu.corpus_bleu(all_srcs, [all_refs]).score
    pooled_chrf = sacrebleu.corpus_chrf(all_srcs, [all_refs]).score
    p_b_ci_lo, p_b_ci_hi = compute_bootstrap_ci(per_utt_bleu)
    p_c_ci_lo, p_c_ci_hi = compute_bootstrap_ci(per_utt_chrf)

    print("\n--------------------------------------------------------------------------------")
    print(f"POOLED MACRO AVERAGE (N={len(all_srcs)}):")
    print(f"  BLEU:   {pooled_bleu:.2f} (95% CI: [{p_b_ci_lo:.2f}, {p_b_ci_hi:.2f}])")
    print(f"  chrF++: {pooled_chrf:.2f} (95% CI: [{p_c_ci_lo:.2f}, {p_c_ci_hi:.2f}])")
    print("================================================================================\n")

    out_file = ROOT_DIR / "data" / "source_copy_baseline_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "per_dialect": results,
            "pooled_bleu": round(pooled_bleu, 2),
            "pooled_chrf": round(pooled_chrf, 2)
        }, f, indent=2, ensure_ascii=False)
    print(f"Source-copy baseline report saved to {out_file}")

if __name__ == "__main__":
    run_source_copy_eval()
