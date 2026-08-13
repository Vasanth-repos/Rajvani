import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "pipeline.yaml"

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "splits": {
            "ratio": [0.8, 0.1, 0.1],
            "test_cap": 500,
            "dev_cap": 300,
            "dev_subsplit": {"promotion": 0.7, "canary": 0.3}
        }
    }

def hash_key_to_float(key: str) -> float:
    """Computes a deterministic float in [0.0, 1.0) from a key string using MD5."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    int_val = int(digest[:8], 16)
    return int_val / 0xFFFFFFFF

def assign_record_split(record: dict, existing_counts: dict = None, config: dict = None):
    """
    Assigns split and dev_subsplit deterministically to a record.
    Rule: Split assignment happens once. If split is already assigned, it is preserved.
    Synthetic records are unconditionally assigned split: train.
    """
    if config is None:
        config = load_config()

    if existing_counts is None:
        existing_counts = {"train": 0, "dev": 0, "test": 0, "dev_promotion": 0, "dev_canary": 0}

    # Synthetic records are assigned 'train' unconditionally
    source = record.get("source", "")
    if source.startswith("synthetic"):
        record["split"] = "train"
        record["dev_subsplit"] = None
        return record

    # If split is already assigned, preserve it
    if record.get("split") in ["train", "dev", "test"]:
        return record

    # Key for hashing
    speaker_id = record.get("speaker_id")
    stable_key = speaker_id if speaker_id else record.get("id", "")
    h = hash_key_to_float(stable_key)

    test_cap = config.get("splits", {}).get("test_cap", 500)
    dev_cap = config.get("splits", {}).get("dev_cap", 300)

    # Base ratio 0.8 train, 0.1 dev, 0.1 test
    if h >= 0.90:
        # Candidate for test
        if existing_counts.get("test", 0) < test_cap:
            split = "test"
        else:
            split = "train"
    elif h >= 0.80:
        # Candidate for dev
        if existing_counts.get("dev", 0) < dev_cap:
            split = "dev"
        else:
            split = "train"
    else:
        split = "train"

    record["split"] = split

    if split == "dev":
        # Determine dev_subsplit (70% promotion, 30% canary)
        h_sub = hash_key_to_float(stable_key + "_sub")
        if h_sub < 0.70:
            record["dev_subsplit"] = "promotion"
        else:
            record["dev_subsplit"] = "canary"
    else:
        record["dev_subsplit"] = None

    return record

def verify_file_path_read_access(file_path: str, caller_script: str):
    """
    Enforces path restrictions:
    training/train_*.py and augmentation/*.py scripts MUST NOT read test.jsonl or dev_canary.jsonl.
    """
    path_str = str(file_path).replace("\\", "/").lower()
    caller_str = str(caller_script).replace("\\", "/").lower()
    is_training_or_aug = ("training/train_" in caller_str or "augmentation/" in caller_str)
    
    if is_training_or_aug:
        if "test.jsonl" in path_str:
            raise PermissionError(f"Access Denied: Script '{caller_script}' is forbidden from reading test split '{file_path}'.")
        if "dev_canary.jsonl" in path_str:
            raise PermissionError(f"Access Denied: Script '{caller_script}' is forbidden from reading dev_canary split '{file_path}'.")

def materialize_splits(dialect: str, input_dir: Path, output_dir: Path):
    """Filter validated records by assigned split and write materialized views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_files = {
        "train": open(output_dir / "train.jsonl", "w", encoding="utf-8"),
        "dev": open(output_dir / "dev.jsonl", "w", encoding="utf-8"),
        "dev_promotion": open(output_dir / "dev_promotion.jsonl", "w", encoding="utf-8"),
        "dev_canary": open(output_dir / "dev_canary.jsonl", "w", encoding="utf-8"),
        "test": open(output_dir / "test.jsonl", "w", encoding="utf-8"),
    }

    counts = {"train": 0, "dev": 0, "test": 0, "dev_promotion": 0, "dev_canary": 0}
    config = load_config()

    validated_files = list(input_dir.glob("*.jsonl"))
    for vfile in validated_files:
        with open(vfile, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec = assign_record_split(rec, counts, config)
                
                s = rec["split"]
                counts[s] = counts.get(s, 0) + 1
                splits_files[s].write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                if s == "dev":
                    sub = rec.get("dev_subsplit")
                    if sub in splits_files:
                        splits_files[sub].write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts[sub] = counts.get(sub, 0) + 1

    for f in splits_files.values():
        f.close()

    print(f"Materialized splits for dialect '{dialect}': {counts}")
    return counts

def main():
    parser = argparse.ArgumentParser(description="Assign splits and materialize split files.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    args = parser.parse_args()

    input_dir = ROOT_DIR / "data" / "validated" / args.dialect
    output_dir = ROOT_DIR / "data" / "splits" / args.dialect
    
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        # Create empty validated file if none exists
        with open(input_dir / "validated_sample.jsonl", "w", encoding="utf-8") as f:
            pass

    materialize_splits(args.dialect, input_dir, output_dir)

if __name__ == "__main__":
    main()
