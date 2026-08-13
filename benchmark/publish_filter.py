import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).parent.parent

K_ANONYMITY_THRESHOLD = 5

def apply_publish_filter(dialect: str = "mwr"):
    test_split_file = ROOT_DIR / "data" / "splits" / dialect / "test.jsonl"
    if not test_split_file.exists():
        print(f"Test split for dialect '{dialect}' missing at {test_split_file}", file=sys.stderr)
        return [], 0, 0

    raw_records = []
    with open(test_split_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    # Consent filter: Keep only public_release_ok: true
    consented_records = [r for r in raw_records if r.get("public_release_ok") is True]
    excluded_consent_count = len(raw_records) - len(consented_records)

    # Map speaker_id to anonymized ordinal
    speaker_map = {}
    speaker_counter = 1

    quasi_counts = Counter()
    for r in consented_records:
        spk = r.get("speaker_id", "unknown_spk")
        if spk not in speaker_map:
            speaker_map[spk] = f"spk_{speaker_counter:03d}"
            speaker_counter += 1
        
        q_key = (r.get("settlement_type", "unknown"), r.get("speaker_age_cohort", "unknown"))
        quasi_counts[q_key] += 1

    filtered_records = []
    k_suppressed_count = 0

    for r in consented_records:
        f_rec = dict(r)
        
        # 1. Strip raw speaker_id & replace with non-reversible ordinal
        orig_spk = f_rec.get("speaker_id", "unknown")
        f_rec["speaker_id"] = speaker_map.get(orig_spk, "spk_000")

        # 2. Generalize district region to dialect level
        f_rec["region"] = dialect.upper()

        # 3. K-anonymity check (k=5)
        q_key = (f_rec.get("settlement_type", "unknown"), f_rec.get("speaker_age_cohort", "unknown"))
        if quasi_counts[q_key] < K_ANONYMITY_THRESHOLD:
            f_rec["settlement_type"] = "unknown"
            f_rec["speaker_age_cohort"] = "unknown"
            k_suppressed_count += 1

        filtered_records.append(f_rec)

    bench_dir = ROOT_DIR / "benchmark" / "published_splits"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_file = bench_dir / f"{dialect}_public_test.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for r in filtered_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Publish filter applied for dialect '{dialect}':")
    print(f"  Raw test records: {len(raw_records)}")
    print(f"  Excluded (public_release_ok: false): {excluded_consent_count}")
    print(f"  Published test records: {len(filtered_records)}")
    print(f"  k-anonymity suppressions (k<{K_ANONYMITY_THRESHOLD}): {k_suppressed_count}")
    print(f"  Saved public benchmark split to: {out_file}")

    return filtered_records, excluded_consent_count, k_suppressed_count

def main():
    parser = argparse.ArgumentParser(description="Filter & anonymize public benchmark test set.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    args = parser.parse_args()

    apply_publish_filter(args.dialect)

if __name__ == "__main__":
    main()
