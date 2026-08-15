"""
eval/eval_nllb_baseline.py

Evaluates zero-shot neural Machine Translation using Meta NLLB-200 (facebook/nllb-200-distilled-600M)
across the 200 held-out test records (data/realworld_test_200.jsonl).
Runs completely blind with zero hand-authored rules or heuristic contamination.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

def run_nllb_baseline_eval(test_path: str = "data/realworld_test_200.jsonl"):
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading neural MT model: {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.2f}s.\n")

    target_lang_id = tokenizer.convert_tokens_to_ids("hin_Deva")

    with open(ROOT_DIR / test_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"=== Running Zero-Shot NLLB-200 Baseline Evaluation (Total Utterances: {len(records)}) ===")

    by_dialect = {}
    all_hyps = []
    all_refs = []
    per_utt_bleu = []
    per_utt_chrf = []

    for idx, r in enumerate(records, 1):
        did = r["dialect"]
        src = r["text_dialect"]
        ref = r.get("reference_hindi", r.get("text_hindi", src))

        inputs = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=128,
                num_beams=2
            )
        hyp = tokenizer.decode(gen[0], skip_special_tokens=True).strip()

        # Sentence-level metrics for CI calculation
        s_bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
        s_chrf = sacrebleu.sentence_chrf(hyp, [ref]).score

        if did not in by_dialect:
            by_dialect[did] = {
                "hyps": [],
                "refs": [],
                "bleu_scores": [],
                "chrf_scores": []
            }

        by_dialect[did]["hyps"].append(hyp)
        by_dialect[did]["refs"].append(ref)
        by_dialect[did]["bleu_scores"].append(s_bleu)
        by_dialect[did]["chrf_scores"].append(s_chrf)

        all_hyps.append(hyp)
        all_refs.append(ref)
        per_utt_bleu.append(s_bleu)
        per_utt_chrf.append(s_chrf)

        if idx % 25 == 0 or idx == len(records):
            print(f"  Processed {idx}/{len(records)} utterances...")

    print("\n================================================================================")
    print("=== RAW ZERO-SHOT NEURAL MT BENCHMARK (NLLB-200 Distilled 600M -> Hindi) ===")
    print("================================================================================")

    results_summary = {}
    for did in sorted(by_dialect.keys()):
        data = by_dialect[did]
        c_bleu = sacrebleu.corpus_bleu(data["hyps"], [data["refs"]]).score
        c_chrf = sacrebleu.corpus_chrf(data["hyps"], [data["refs"]]).score
        b_ci_lo, b_ci_hi = compute_bootstrap_ci(data["bleu_scores"])
        c_ci_lo, c_ci_hi = compute_bootstrap_ci(data["chrf_scores"])

        results_summary[did] = {
            "n": len(data["hyps"]),
            "bleu": round(c_bleu, 2),
            "bleu_ci": [b_ci_lo, b_ci_hi],
            "chrf": round(c_chrf, 2),
            "chrf_ci": [c_ci_lo, c_ci_hi]
        }

        print(f"Dialect {did.upper()} (N={len(data['hyps'])}):")
        print(f"  BLEU:   {c_bleu:.2f} (95% CI: [{b_ci_lo:.2f}, {b_ci_hi:.2f}])")
        print(f"  chrF++: {c_chrf:.2f} (95% CI: [{c_ci_lo:.2f}, {c_ci_hi:.2f}])")

    pooled_bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score
    pooled_chrf = sacrebleu.corpus_chrf(all_hyps, [all_refs]).score
    p_b_ci_lo, p_b_ci_hi = compute_bootstrap_ci(per_utt_bleu)
    p_c_ci_lo, p_c_ci_hi = compute_bootstrap_ci(per_utt_chrf)

    print("\n--------------------------------------------------------------------------------")
    print(f"POOLED MACRO AVERAGE (N={len(all_hyps)}):")
    print(f"  BLEU:   {pooled_bleu:.2f} (95% CI: [{p_b_ci_lo:.2f}, {p_b_ci_hi:.2f}])")
    print(f"  chrF++: {pooled_chrf:.2f} (95% CI: [{p_c_ci_lo:.2f}, {p_c_ci_hi:.2f}])")
    print("================================================================================\n")

    # Save raw audit snapshot
    out_file = ROOT_DIR / "data" / "nllb_zero_shot_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "target_language": "hin_Deva",
            "per_dialect": results_summary,
            "pooled_bleu": round(pooled_bleu, 2),
            "pooled_chrf": round(pooled_chrf, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2, ensure_ascii=False)
    print(f"Evaluation report saved to {out_file}")

if __name__ == "__main__":
    run_nllb_baseline_eval()
