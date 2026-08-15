import json
import sys
import sacrebleu
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.stdout.reconfigure(encoding='utf-8')

from serving.mt_engine.rajasthani_mt import translate_dialect_to_hindi

def evaluate_empirical_mt(test_path: str = "data/realworld_test_200.jsonl"):
    with open(ROOT_DIR / test_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    by_dialect = {}
    for r in records:
        d = r["dialect"]
        src = r["text_dialect"]
        ref = r.get("reference_hindi", src)
        hyp = translate_dialect_to_hindi(src, dialect=d)
        
        if d not in by_dialect:
            by_dialect[d] = {"hyps": [], "refs": [], "samples": []}
            
        by_dialect[d]["hyps"].append(hyp)
        by_dialect[d]["refs"].append(ref)
        if len(by_dialect[d]["samples"]) < 2:
            by_dialect[d]["samples"].append({"src": src, "hyp": hyp, "ref": ref})

    results = {}
    all_hyps = []
    all_refs = []

    print("================================================================================")
    print("=== EMPIRICAL MT EVALUATION: RAJASTHANI -> STANDARD HINDI (N=200 TEST SET) ===")
    print("================================================================================")

    for d in sorted(by_dialect.keys()):
        data = by_dialect[d]
        bleu = sacrebleu.corpus_bleu(data["hyps"], [data["refs"]]).score
        chrf = sacrebleu.corpus_chrf(data["hyps"], [data["refs"]]).score
        
        results[d] = {
            "sample_count": len(data["hyps"]),
            "bleu": round(bleu, 2),
            "chrf": round(chrf, 2),
            "samples": data["samples"]
        }
        all_hyps.extend(data["hyps"])
        all_refs.extend(data["refs"])

        print(f"\n--- Dialect {d.upper()} (N={len(data['hyps'])}) ---")
        print(f"  BLEU:   {bleu:.2f}")
        print(f"  chrF++: {chrf:.2f}")
        for idx, s in enumerate(data["samples"], 1):
            print(f"  Sample {idx}:")
            print(f"    Source (Dialect): {s['src']}")
            print(f"    Hypothesis (MT):  {s['hyp']}")
            print(f"    Reference (Gold): {s['ref']}")

    total_bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score
    total_chrf = sacrebleu.corpus_chrf(all_hyps, [all_refs]).score

    print("\n================================================================================")
    print(f"=== POOLED EMPIRICAL MT PERFORMANCE (N={len(all_hyps)}): BLEU = {total_bleu:.2f} | chrF++ = {total_chrf:.2f} ===")
    print("================================================================================")

    return results, round(total_bleu, 2), round(total_chrf, 2)

if __name__ == "__main__":
    evaluate_empirical_mt()
