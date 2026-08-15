"""
eval/verify_leakage.py

Automated dataset split isolation and leakage verification.
Guarantees zero overlap in record IDs and raw text pairs between
train/dev/augmented pools and held-out test splits.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Set

ROOT_DIR = Path(__file__).parent.parent
DIALECTS = ["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"]

def load_ids_and_texts(file_path: Path) -> Tuple[Set[str], Set[str]]:
    ids = set()
    texts = set()
    if not file_path.exists():
        return ids, texts
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    if "id" in r:
                        ids.add(str(r["id"]))
                    elif "audio_id" in r:
                        ids.add(str(r["audio_id"]))
                    if "text_dialect" in r and r["text_dialect"]:
                        texts.add(r["text_dialect"].strip())
                    elif "text" in r and r["text"]:
                        texts.add(r["text"].strip())
                except Exception:
                    continue
    return ids, texts

def verify_all_splits() -> Dict[str, Any]:
    results = {}
    total_leaks = 0

    for d in DIALECTS:
        test_path = ROOT_DIR / "data" / "splits" / d / "test.jsonl"
        # If test.jsonl is empty or split is realworld_test_200, check realworld_test_200.jsonl
        test_ids, test_texts = load_ids_and_texts(test_path)
        
        # Also check against regional subset of realworld_test_200.jsonl
        rw_path = ROOT_DIR / "data" / "realworld_test_200.jsonl"
        if rw_path.exists():
            with open(rw_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("dialect", "").lower() == d:
                            if "id" in r:
                                test_ids.add(str(r["id"]))
                            if "text_dialect" in r and r["text_dialect"]:
                                test_texts.add(r["text_dialect"].strip())

        leak_targets = [
            ROOT_DIR / "data" / "splits" / d / "train.jsonl",
            ROOT_DIR / "data" / "splits" / d / "dev.jsonl",
            ROOT_DIR / "data" / "splits" / d / "dev_canary.jsonl",
            ROOT_DIR / "data" / "splits" / d / "dev_promotion.jsonl",
            ROOT_DIR / "data" / "synthetic" / d / "backtranslation.jsonl",
            ROOT_DIR / "data" / "synthetic" / f"{d}_augmented.jsonl",
            ROOT_DIR / "data" / "processed" / f"{d}_normalized.jsonl",
        ]

        dialect_leak_info = []
        for target in leak_targets:
            if target.exists():
                t_ids, t_texts = load_ids_and_texts(target)
                id_overlap = test_ids & t_ids
                text_overlap = test_texts & t_texts
                leak_count = len(id_overlap) + len(text_overlap)
                total_leaks += leak_count
                dialect_leak_info.append({
                    "target_file": str(target.relative_to(ROOT_DIR)),
                    "id_overlap": len(id_overlap),
                    "text_overlap": len(text_overlap),
                    "is_clean": (leak_count == 0)
                })

        results[d] = {
            "test_sample_count": len(test_texts),
            "targets_checked": dialect_leak_info,
            "leak_free": all(item["is_clean"] for item in dialect_leak_info)
        }

    return {
        "status": "PASS" if total_leaks == 0 else "FAIL",
        "total_leak_count": total_leaks,
        "dialects": results
    }

if __name__ == "__main__":
    report = verify_all_splits()
    print(f"=== Dataset Split Leakage Audit: {report['status']} (Total Overlaps Detected: {report['total_leak_count']}) ===")
    for d, info in report["dialects"].items():
        status_icon = "[OK]" if info["leak_free"] else "[FAIL]"
        print(f"\n  {status_icon} Dialect {d.upper()} (Test Set: {info['test_sample_count']} utterances audited):")
        for pool in info["targets_checked"]:
            pool_status = "[CLEAN]" if pool["is_clean"] else "[LEAK DETECTED]"
            print(f"    - {pool['target_file']:<45} : id_overlap={pool['id_overlap']}, text_overlap={pool['text_overlap']} {pool_status}")
