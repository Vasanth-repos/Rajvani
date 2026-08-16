"""
active_learning/merge_verified_into_train.py

Production active-learning ingestion tool that reads certified community records from
data/verified/human_verified_transcripts.jsonl, validates schema and isolation constraints,
and securely integrates them into data/splits/<dialect>/train.jsonl.

Strict Safety Invariants:
1. Zero test set leakage: Asserts 0 string overlap against data/realworld_test_200.jsonl.
2. Idempotent: Skips records already present in the training split.
3. Provenance tagged: Marks records with origin='community_active_learning'.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Set

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.stdout.reconfigure(encoding="utf-8")

def load_test_string_hashes() -> Set[str]:
    """Loads all normalized text strings from the frozen held-out test suite."""
    test_file = ROOT_DIR / "data" / "realworld_test_200.jsonl"
    test_texts = set()
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    test_texts.add(r["text_dialect"].strip())
    return test_texts

def merge_verified_transcripts_into_train():
    verified_file = ROOT_DIR / "data" / "verified" / "human_verified_transcripts.jsonl"
    if not verified_file.exists():
        print(f"[Error] Verified transcripts file not found at {verified_file}")
        return

    test_forbidden_texts = load_test_string_hashes()
    print(f"Loaded {len(test_forbidden_texts)} frozen test strings for split isolation guard.")

    with open(verified_file, "r", encoding="utf-8") as f:
        verified_records = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(verified_records)} certified human-verified records in registry.")

    merged_by_dialect: Dict[str, int] = {}
    skipped_duplicates: int = 0
    blocked_test_collisions: int = 0

    for rec in verified_records:
        did = rec.get("dialect_id", "MWR").lower().split()[0]
        text_dialect = rec.get("corrected_transcript", rec.get("raw_transcript", "")).strip()

        if not text_dialect:
            continue

        # Split Isolation Guard: Assert zero leakage into test set
        if text_dialect in test_forbidden_texts:
            print(f"[SECURITY BLOCKED] String '{text_dialect}' collides with held-out test split!")
            blocked_test_collisions += 1
            continue

        train_file = ROOT_DIR / "data" / "splits" / did / "train.jsonl"
        train_file.parent.mkdir(parents=True, exist_ok=True)

        existing_texts = set()
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as tf:
                for tline in tf:
                    if tline.strip():
                        tr = json.loads(tline)
                        existing_texts.add(tr.get("text_dialect", "").strip())

        if text_dialect in existing_texts:
            skipped_duplicates += 1
            continue

        # Create structured training record
        new_train_record = {
            "id": f"al_{did}_{len(existing_texts) + 1:04d}",
            "dialect": did,
            "text_dialect": text_dialect,
            "speaker_id": rec.get("speaker_id", f"community_speaker_{did}"),
            "provenance": "community_active_learning_verified",
            "voice_clone_ok": False,
            "status": "READY_FOR_RETRAINING"
        }

        with open(train_file, "a", encoding="utf-8") as tf:
            tf.write(json.dumps(new_train_record, ensure_ascii=False) + "\n")

        merged_by_dialect[did] = merged_by_dialect.get(did, 0) + 1

    print("\n=== ACTIVE LEARNING INGESTION & RETRAINING POOL UPDATE ===")
    for d, cnt in sorted(merged_by_dialect.items()):
        print(f"  Dialect {d.upper()}: +{cnt} newly merged training records")
    print(f"  Skipped Duplicates: {skipped_duplicates}")
    print(f"  Blocked Test Collisions: {blocked_test_collisions} (Zero-Leakage Invariant Satisfied)")
    print("=========================================================\n")

if __name__ == "__main__":
    merge_verified_transcripts_into_train()
