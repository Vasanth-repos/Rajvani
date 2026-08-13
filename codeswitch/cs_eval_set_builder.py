import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from codeswitch.tagger import process_record_codeswitching

def build_code_switched_eval_sets(dialect: str):
    splits_dir = ROOT_DIR / "data" / "splits" / dialect
    if not splits_dir.exists():
        print(f"Splits directory {splits_dir} does not exist.", file=sys.stderr)
        return

    test_file = splits_dir / "test.jsonl"
    if not test_file.exists():
        print(f"Test split file {test_file} does not exist.", file=sys.stderr)
        return

    all_records = []
    cs_records = []
    mono_records = []

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec = process_record_codeswitching(rec)
                all_records.append(rec)
                if rec.get("is_code_switched"):
                    cs_records.append(rec)
                else:
                    mono_records.append(rec)

    cs_share = (len(cs_records) / len(all_records)) if all_records else 0.0

    # Write separate subset files
    cs_out = splits_dir / "test_codeswitched.jsonl"
    mono_out = splits_dir / "test_monolingual.jsonl"

    with open(cs_out, "w", encoding="utf-8") as f:
        for r in cs_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(mono_out, "w", encoding="utf-8") as f:
        for r in mono_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Dialect '{dialect}' Test Split Analysis:")
    print(f"  Total test records: {len(all_records)}")
    print(f"  Code-switched count: {len(cs_records)} ({cs_share*100:.1f}%)")
    print(f"  Monolingual count: {len(mono_records)}")
    print(f"  Created {cs_out.name} and {mono_out.name}")

def main():
    parser = argparse.ArgumentParser(description="Build code-switched eval subsets for dev/test splits.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    args = parser.parse_args()

    build_code_switched_eval_sets(args.dialect)

if __name__ == "__main__":
    main()
