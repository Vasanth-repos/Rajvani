"""
training/run_all_finetuning.py

Executes end-to-end fine-tuning across all 6 Rajasthani dialects (Marwari, Mewari, Dhundhari, Hadoti, Mewati, Bagri)
for Machine Translation (IndicTrans2 LoRA), ASR (Whisper-v3-Turbo LoRA), TTS (Meta MMS-TTS), and Dialect-ID Classification.
"""

import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from configs.dialects import DIALECT_REGISTRY
from training.train_mt import run_mt_training
from training.train_asr import run_asr_training
from training.train_tts import run_tts_training
from dialect_id.train import train_dialect_id_classifier
from eval.eval_realworld_200 import run_realworld_benchmark

def main():
    print("=" * 70)
    print("🚀 STARTING RAJVANI FULL-DIALECT FINE-TUNING PIPELINE")
    print("=" * 70)
    start_time = time.time()
    
    dialects = list(DIALECT_REGISTRY.keys())
    results_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dialects": dialects,
        "mt_runs": {},
        "asr_runs": {},
        "tts_runs": {},
        "dialect_id": None,
        "eval_benchmark": None
    }

    # 1. Dialect-ID Classifier Fine-Tuning
    print("\n--- [Phase 1/4] Fine-Tuning Dialect-ID Classifier ---")
    did_out = train_dialect_id_classifier()
    results_summary["dialect_id"] = str(did_out)

    # 2. Machine Translation (MT) LoRA Fine-Tuning (IndicTrans2-1B)
    print("\n--- [Phase 2/4] Fine-Tuning MT LoRA Adapters (6 Dialects) ---")
    for d in dialects:
        did = d.lower()
        print(f"\n[MT] Fine-tuning {d} ({DIALECT_REGISTRY[d]['name']})...")
        run_id = run_mt_training(dialect=did, target_lang="hin", epochs=5)
        results_summary["mt_runs"][d] = run_id

    # 3. ASR LoRA Fine-Tuning (Whisper-v3-Turbo)
    print("\n--- [Phase 3/4] Fine-Tuning ASR LoRA Adapters (6 Dialects) ---")
    for d in dialects:
        did = d.lower()
        print(f"\n[ASR] Fine-tuning {d} ({DIALECT_REGISTRY[d]['name']})...")
        run_id = run_asr_training(dialect=did, epochs=5)
        results_summary["asr_runs"][d] = run_id

    # 4. Text-to-Speech (TTS) Meta MMS-TTS Fine-Tuning
    print("\n--- [Phase 4/4] Fine-Tuning TTS Voice Checkpoints ---")
    for d in dialects:
        did = d.lower()
        print(f"\n[TTS] Fine-tuning {d} ({DIALECT_REGISTRY[d]['name']})...")
        run_id = run_tts_training(dialect=did, backend="mms", epochs=5)
        results_summary["tts_runs"][d] = run_id

    # 5. Real-World Benchmark Evaluation
    print("\n--- [Evaluation] Evaluating Fine-Tuned Checkpoints Against Real-World Test Suite ---")
    eval_metrics = run_realworld_benchmark(mode="finetuned")
    results_summary["eval_benchmark"] = eval_metrics

    # Save summary report
    out_file = ROOT_DIR / "data" / "finetuning_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"🎉 ALL MODEL FINE-TUNING AND EVALUATION COMPLETED IN {elapsed}s")
    print(f"Summary saved to: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
