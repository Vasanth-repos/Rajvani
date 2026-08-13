import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).parent.parent

def generate_augmentation_report(dialect: str = "all"):
    synthetic_dir = ROOT_DIR / "data" / "synthetic"
    validated_dir = ROOT_DIR / "data" / "validated"

    dialects = [dialect] if dialect != "all" else ["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"]

    print("=== Synthetic Data Augmentation Audit Report ===")
    for d in dialects:
        d_synth = synthetic_dir / d
        d_val = validated_dir / d

        # Count validated real pairs/hours
        real_text_count = 0
        real_audio_hours = 0.0
        if d_val.exists():
            for fpath in d_val.glob("*.jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            if rec.get("audio_path"):
                                real_audio_hours += rec.get("duration_sec", 0.0) / 3600.0
                            else:
                                real_text_count += 1

        # Count synthetic records
        synth_counts = Counter()
        checkpoint_dist = Counter()
        base_checkpoint_non_superseded = 0
        synth_tts_hours = 0.0

        if d_synth.exists():
            for fpath in d_synth.glob("*.jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            src = rec.get("source", "unknown")
                            synth_counts[src] += 1

                            if src == "synthetic_tts":
                                synth_tts_hours += rec.get("duration_sec", 0.0) / 3600.0

                            if src == "synthetic_backtranslation":
                                is_superseded = rec.get("superseded", False)
                                chk = rec.get("generator_checkpoint", "base")
                                if not is_superseded:
                                    checkpoint_dist[chk] += 1
                                    if chk == "base":
                                        base_checkpoint_non_superseded += 1

        # Check caps
        total_audio_hours = real_audio_hours + synth_tts_hours
        tts_share = (synth_tts_hours / total_audio_hours) if total_audio_hours > 0 else 0.0
        tts_cap_ok = tts_share <= 0.30

        print(f"\nDialect: {d.upper()}")
        print(f"  Real text records: {real_text_count} | Real audio hours: {real_audio_hours:.2f}h")
        print(f"  Synthetic Breakdown: {dict(synth_counts)}")
        print(f"  TTS Share of Total ASR Hours: {tts_share*100:.1f}% (Cap <=30%: {'PASS' if tts_cap_ok else 'EXCEEDED'})")
        print(f"  Back-translation Checkpoint Distribution (Active Pool): {dict(checkpoint_dist)}")
        print(f"  Active 'base' Checkpoint BT Count: {base_checkpoint_non_superseded}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmentation pool audit report.")
    parser.add_argument("--dialect", type=str, default="all", help="Dialect ID or 'all'")
    args = parser.parse_args()

    generate_augmentation_report(args.dialect)
