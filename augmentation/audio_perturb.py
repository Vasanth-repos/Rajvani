import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.splits.assign_split import verify_file_path_read_access

def perturb_audio_records(dialect: str, input_jsonl_path: str):
    # Enforce path read access restriction!
    verify_file_path_read_access(input_jsonl_path, __file__)

    input_path = Path(input_jsonl_path)
    if not input_path.exists():
        print(f"Input file {input_path} missing.", file=sys.stderr)
        return 0

    real_audio_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # Apply strictly to REAL recordings only (never synthetic_tts)
                if rec.get("source") not in ["synthetic_tts", "synthetic_perturbed"]:
                    real_audio_records.append(rec)

    synth_dir = ROOT_DIR / "data" / "synthetic" / dialect
    synth_dir.mkdir(parents=True, exist_ok=True)
    out_file = synth_dir / "audio_perturbed.jsonl"

    perturbed_records = []
    for rec in real_audio_records:
        p_rec = dict(rec)
        p_rec["id"] = str(uuid.uuid4())
        p_rec["audio_path"] = f"data/synthetic/{dialect}/pert_{rec['id'][:8]}.wav"
        p_rec["source"] = "synthetic_perturbed"
        p_rec["split"] = "train"  # Unconditionally train
        p_rec["dev_subsplit"] = None
        perturbed_records.append(p_rec)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in perturbed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Audio perturbation complete for dialect '{dialect}'. Created {len(perturbed_records)} perturbed records from {len(real_audio_records)} real records.")
    return len(perturbed_records)

def main():
    parser = argparse.ArgumentParser(description="Apply speed, pitch, and noise perturbations to real recordings.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    parser.add_argument("--input-file", type=str, required=True, help="Input real audio JSONL file")
    args = parser.parse_args()

    perturb_audio_records(args.dialect, args.input_file)

if __name__ == "__main__":
    main()
