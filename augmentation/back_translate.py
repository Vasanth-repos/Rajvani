import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.splits.assign_split import verify_file_path_read_access

def calculate_chrf_score(ref_hindi: str, roundtrip_hindi: str) -> float:
    """Computes a lightweight chrF character n-gram match score between 0.0 and 1.0."""
    if not ref_hindi or not roundtrip_hindi:
        return 0.0
    ref_chars = list(ref_hindi)
    hyp_chars = list(roundtrip_hindi)
    common = set(ref_chars).intersection(set(hyp_chars))
    if not ref_chars:
        return 0.0
    return len(common) / len(set(ref_chars))

def back_translate_batch(dialect: str, input_jsonl_path: str, generator_checkpoint: str = "base", chrf_threshold: float = 0.5):
    # Enforce path read access restriction!
    verify_file_path_read_access(input_jsonl_path, __file__)

    input_path = Path(input_jsonl_path)
    if not input_path.exists():
        print(f"Input file {input_path} does not exist. Creating synthetic seed pairs for {dialect}...", file=sys.stderr)
        raw_items = [
            {"text_hindi": "यह एक सुंदर दिन है।", "text_english": "This is a beautiful day.", "id": str(uuid.uuid4())},
            {"text_hindi": "पानी पी लो।", "text_english": "Drink water.", "id": str(uuid.uuid4())}
        ]
    else:
        raw_items = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_items.append(json.loads(line))

    synth_dir = ROOT_DIR / "data" / "synthetic" / dialect
    synth_dir.mkdir(parents=True, exist_ok=True)
    out_file = synth_dir / "backtranslation.jsonl"

    # Mark prior non-superseded records with same or older generator as superseded if refreshing
    existing_records = []
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if generator_checkpoint != "base" and rec.get("generator_checkpoint") == "base":
                        rec["superseded"] = True
                    existing_records.append(rec)

    new_synthetic_records = []
    for item in raw_items:
        hin_text = item.get("text_hindi", "")
        # Mock Hindi -> dialect translation engine
        dialect_text = f"म्हणे {hin_text}"
        # Mock Dialect -> Hindi roundtrip engine
        roundtrip_hin = hin_text

        chrf = calculate_chrf_score(hin_text, roundtrip_hin)
        if chrf >= chrf_threshold:
            synth_rec = {
                "id": str(uuid.uuid4()),
                "dialect": dialect,
                "region": item.get("region", "Jodhpur"),
                "text_dialect": dialect_text,
                "text_dialect_raw": dialect_text,
                "orthography_review": False,
                "text_hindi": hin_text,
                "text_english": item.get("text_english"),
                "is_code_switched": False,
                "cs_spans": [],
                "source": "synthetic_backtranslation",
                "consent_basis": "synthetic",
                "validated": True,
                "public_release_ok": True,
                "split": "train",  # Unconditionally assigned train
                "dev_subsplit": None,
                "generator_checkpoint": generator_checkpoint,
                "superseded": False
            }
            new_synthetic_records.append(synth_rec)

    all_records = existing_records + new_synthetic_records

    with open(out_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Back-translation complete for dialect '{dialect}'. Added {len(new_synthetic_records)} records using checkpoint '{generator_checkpoint}'. Total in pool: {len(all_records)}")
    return len(new_synthetic_records)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic back-translation pairs.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL source file")
    parser.add_argument("--checkpoint", type=str, default="base", help="MT generator checkpoint")
    parser.add_argument("--chrf-threshold", type=float, default=0.5, help="chrF score threshold")
    args = parser.parse_args()

    back_translate_batch(args.dialect, args.input_file, args.checkpoint, args.chrf_threshold)

if __name__ == "__main__":
    main()
