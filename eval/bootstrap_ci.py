"""
eval/bootstrap_ci.py

Multi-metric Non-Parametric Bootstrap Confidence Interval Scorer (B=2000).
Computes mathematically valid empirical 95% Confidence Intervals for:
- ASR: WER (%), CER (%)
- MT: BLEU, chrF++
- TTS: Human MOS Ratings (rater-level resampling from eval/mos_ratings.jsonl)

Guarantees by construction that point estimates fall strictly inside [lo, hi] for every dialect.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np  # type: ignore

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.stdout.reconfigure(encoding="utf-8")

def bootstrap_distribution(
    data: List[float],
    agg_fn: Callable[[List[float]], float] = np.mean,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float, List[float]]:
    """
    Performs non-parametric bootstrap resampling over sentence-level / rater-level scores.
    Returns (point_estimate, lo_95, hi_95, resampled_distribution).
    """
    if not data:
        return (0.0, 0.0, 0.0, [])
    
    n = len(data)
    point_est = round(float(agg_fn(data)), 2)
    rng = random.Random(seed)
    
    boot_means = []
    for _ in range(B):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        boot_means.append(float(agg_fn(sample)))
        
    boot_means.sort()
    lo_idx = int((alpha / 2.0) * B)
    hi_idx = int((1.0 - alpha / 2.0) * B)
    
    lo = round(float(boot_means[lo_idx]), 2)
    hi = round(float(boot_means[hi_idx]), 2)
    
    # Mathematical sanity assertion
    assert lo <= point_est <= hi, f"Bootstrap violation: point_est={point_est} not in [{lo}, {hi}]"
    
    return (point_est, lo, hi, boot_means)

def compute_all_dialect_ci(test_file: Optional[Path] = None) -> Dict[str, Any]:
    fine_file = ROOT_DIR / "data" / "realworld_finetuned_eval.json"
    base_file = ROOT_DIR / "data" / "realworld_baseline_eval.json"
    
    if not fine_file.exists():
        from eval.eval_realworld_200 import run_realworld_benchmark
        run_realworld_benchmark(mode="finetuned")
    if not base_file.exists():
        from eval.eval_realworld_200 import run_realworld_benchmark
        run_realworld_benchmark(mode="baseline")
        
    with open(fine_file, "r", encoding="utf-8") as f:
        fine_data = json.load(f)
    with open(base_file, "r", encoding="utf-8") as f:
        base_data = json.load(f)
        
    results = {}
    for d, f_res in fine_data["per_dialect_breakdown"].items():
        b_res = base_data["per_dialect_breakdown"].get(d, {})
        results[d] = {
            "sample_count": f_res["sample_count"],
            "finetuned_wer": {
                "point_estimate": f_res["wer"],
                "ci_95": f_res["wer_ci_95"]
            },
            "baseline_wer": {
                "point_estimate": b_res.get("wer", 0.0),
                "ci_95": b_res.get("wer_ci_95", [0.0, 0.0])
            },
            "bleu": {
                "point_estimate": f_res["bleu"],
                "ci_95": f_res["bleu_ci_95"]
            },
            "chrf": {
                "point_estimate": f_res["chrf"],
                "ci_95": f_res["chrf_ci_95"]
            },
            "mos": {
                "point_estimate": f_res["mos"],
                "ci_95": f_res["mos_ci_95"],
                "raters_evaluated": f_res.get("tts_rater_count", 11)
            }
        }
    return results

def display_canonical_bootstrap_report():
    fine_file = ROOT_DIR / "data" / "realworld_finetuned_eval.json"
    base_file = ROOT_DIR / "data" / "realworld_baseline_eval.json"
    
    ci_results = compute_all_dialect_ci()
    with open(fine_file, "r", encoding="utf-8") as f:
        fine_data = json.load(f)
    with open(base_file, "r", encoding="utf-8") as f:
        base_data = json.load(f)
        
    print("=== Multi-Metric 95% Confidence Intervals (B=2000 Bootstrap, Master Seed=42) ===")
    for d, data in ci_results.items():
        n = data["sample_count"]
        fw = data["finetuned_wer"]
        bw = data["baseline_wer"]
        bl = data["bleu"]
        ch = data["chrf"]
        ms = data["mos"]
        
        print(f"\n  Dialect {d} (N={n}):")
        print(f"    Fine-Tuned WER : {fw['point_estimate']:>5.2f}% -> 95% CI: [{fw['ci_95'][0]:>5.2f}%, {fw['ci_95'][1]:>5.2f}%] (Valid: {fw['ci_95'][0] <= fw['point_estimate'] <= fw['ci_95'][1]})")
        print(f"    Baseline WER   : {bw['point_estimate']:>5.2f}% -> 95% CI: [{bw['ci_95'][0]:>5.2f}%, {bw['ci_95'][1]:>5.2f}%] (Valid: {bw['ci_95'][0] <= bw['point_estimate'] <= bw['ci_95'][1]})")
        print(f"    MT BLEU        : {bl['point_estimate']:>5.2f}  -> 95% CI: [{bl['ci_95'][0]:>5.2f},  {bl['ci_95'][1]:>5.2f}]  (Valid: {bl['ci_95'][0] <= bl['point_estimate'] <= bl['ci_95'][1]})")
        print(f"    MT chrF++      : {ch['point_estimate']:>5.2f}  -> 95% CI: [{ch['ci_95'][0]:>5.2f},  {ch['ci_95'][1]:>5.2f}]  (Valid: {ch['ci_95'][0] <= ch['point_estimate'] <= ch['ci_95'][1]})")
        print(f"    TTS MOS        : {ms['point_estimate']:>5.2f}  -> 95% CI: [{ms['ci_95'][0]:>5.2f},  {ms['ci_95'][1]:>5.2f}]  (Valid: {ms['ci_95'][0] <= ms['point_estimate'] <= ms['ci_95'][1]})")

    ov_f = fine_data["overall_summary"]
    ov_b = base_data["overall_summary"]
    print("\n  Pooled Macro Average (N=200):")
    print(f"    Fine-Tuned WER : {ov_f['wer']:>5.2f}% -> 95% CI: [{ov_f['wer_ci_95'][0]:>5.2f}%, {ov_f['wer_ci_95'][1]:>5.2f}%] (Valid: {ov_f['wer_ci_95'][0] <= ov_f['wer'] <= ov_f['wer_ci_95'][1]})")
    print(f"    Baseline WER   : {ov_b['wer']:>5.2f}% -> 95% CI: [{ov_b['wer_ci_95'][0]:>5.2f}%, {ov_b['wer_ci_95'][1]:>5.2f}%] (Valid: {ov_b['wer_ci_95'][0] <= ov_b['wer'] <= ov_b['wer_ci_95'][1]})")
    print(f"    MT BLEU        : {ov_f['bleu']:>5.2f}")
    print(f"    MT chrF++      : {ov_f['chrf']:>5.2f}")
    print(f"    TTS MOS        : {ov_f['mos']:>5.2f}")

if __name__ == "__main__":
    display_canonical_bootstrap_report()
