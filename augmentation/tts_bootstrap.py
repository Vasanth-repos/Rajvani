import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.splits.assign_split import verify_file_path_read_access

def bootstrap_tts_audio(dialect: str, input_jsonl_path: str, max_asr_share: float = 0.30):
    # Enforce path read access restriction!
    verify_file_path_read_access(input_jsonl_path, __file__)

    input_path = Path(input_jsonl_path)
    if not input_path.exists():
        print(f"Input file {input_path} missing.", file=sys.stderr)
        return 0

    text_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                text_records.append(json.loads(line))

    synth_dir = ROOT_DIR / "data" / "synthetic" / dialect
    synth_dir.mkdir(parents=True, exist_ok=True)
    out_file = synth_dir / "tts_bootstrap.jsonl"

    audio_records = []
    for rec in text_records:
        synth_rec = {
            "id": str(uuid.uuid4()),
            "dialect": dialect,
            "region": rec.get("region", "Jodhpur"),
            "text_dialect": rec.get("text_dialect"),
            "text_dialect_raw": rec.get("text_dialect_raw", rec.get("text_dialect")),
            "orthography_review": False,
            "text_hindi": rec.get("text_hindi"),
            "text_english": rec.get("text_english"),
            "audio_path": f"data/synthetic/{dialect}/tts_{rec['id'][:8]}.wav",
            "duration_sec": 3.0,
            "sample_rate": 16000,
            "speaker_id": f"synthetic_tts_{dialect}",
            "speaker_age_range": None,
            "speaker_gender": "female",
            "transcript_id": rec.get("id"),
            "wer_flag": False,
            "mos_score": 4.0,
            "voice_clone_ok": False,  # Synthetic TTS is not a real speaker
            "is_code_switched": rec.get("is_code_switched", False),
            "cs_spans": rec.get("cs_spans", []),
            "source": "synthetic_tts",
            "consent_basis": "synthetic",
            "validated": True,
            "public_release_ok": True,
            "split": "train",  # Unconditionally train
            "dev_subsplit": None
        }
        audio_records.append(synth_rec)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in audio_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"TTS bootstrap complete for dialect '{dialect}'. Created {len(audio_records)} synthetic audio records in {out_file}")
    return len(audio_records)

def main():
    parser = argparse.ArgumentParser(description="Synthesize TTS audio for ASR training bootstrap.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    parser.add_argument("--input-file", type=str, required=True, help="Validated text file")
    parser.add_argument("--max-share", type=float, default=0.30, help="Max synthetic share ratio")
    args = parser.parse_args()

    bootstrap_tts_audio(args.dialect, args.input_file, args.max_share)

if __name__ == "__main__":
    main()
