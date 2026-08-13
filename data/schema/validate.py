import argparse
import json
import os
import sys
import uuid
from pathlib import Path
import jsonschema

SCHEMA_DIR = Path(__file__).parent
TEXT_SCHEMA_PATH = SCHEMA_DIR / "text_record.schema.json"
AUDIO_SCHEMA_PATH = SCHEMA_DIR / "audio_record.schema.json"

def load_schemas():
    with open(TEXT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        text_schema = json.load(f)
    with open(AUDIO_SCHEMA_PATH, "r", encoding="utf-8") as f:
        audio_schema = json.load(f)
    return text_schema, audio_schema

def create_sample_text_record(dialect="mwr"):
    return {
        "id": str(uuid.uuid4()),
        "dialect": dialect,
        "region": "Jodhpur",
        "text_dialect": "म्हारो नाम राम है।",
        "text_dialect_raw": "महारो नाम राम है।",
        "orthography_review": False,
        "text_hindi": "मेरा नाम राम है।",
        "text_english": "My name is Ram.",
        "is_code_switched": False,
        "cs_spans": [],
        "source": "field_collection",
        "consent_basis": "explicit_written",
        "validated": True,
        "validator_id": "val_01",
        "confidence_score": 0.95,
        "speaker_age_cohort": "31-50",
        "settlement_type": "rural",
        "public_release_ok": True,
        "split": "train",
        "dev_subsplit": None
    }

def create_sample_audio_record(dialect="mwr"):
    return {
        "id": str(uuid.uuid4()),
        "dialect": dialect,
        "region": "Jodhpur",
        "text_dialect": "म्हारो नाम राम है।",
        "text_dialect_raw": "महारो नाम राम है।",
        "orthography_review": False,
        "text_hindi": "मेरा नाम राम है।",
        "text_english": "My name is Ram.",
        "audio_path": f"data/raw/{dialect}/sample_01.wav",
        "duration_sec": 3.5,
        "sample_rate": 16000,
        "speaker_id": "spk_101",
        "speaker_age_range": "30-40",
        "speaker_gender": "male",
        "transcript_id": str(uuid.uuid4()),
        "wer_flag": False,
        "mos_score": 4.2,
        "voice_clone_ok": False,
        "is_code_switched": False,
        "cs_spans": [],
        "source": "field_collection",
        "consent_basis": "explicit_written",
        "validated": True,
        "validator_id": "val_01",
        "confidence_score": 0.95,
        "speaker_age_cohort": "31-50",
        "settlement_type": "rural",
        "public_release_ok": False,
        "split": "train",
        "dev_subsplit": None
    }

def validate_record(record, record_type="text"):
    text_schema, audio_schema = load_schemas()
    schema = text_schema if record_type == "text" else audio_schema
    jsonschema.validate(instance=record, schema=schema)
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate data records against JSON schemas.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID to validate")
    parser.add_argument("--file", type=str, help="Path to json/jsonl file to validate")
    parser.add_argument("--type", type=str, choices=["text", "audio"], default="text", help="Record type")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: File {path} does not exist.", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                validate_record(rec, args.type)
            except jsonschema.ValidationError as e:
                print(f"Validation error on line {idx+1} in {path}: {e.message}", file=sys.stderr)
                sys.exit(1)
        print(f"Successfully validated {len(lines)} records from {path}")
    else:
        # Validate sample records
        sample_text = create_sample_text_record(args.dialect)
        sample_audio = create_sample_audio_record(args.dialect)
        validate_record(sample_text, "text")
        validate_record(sample_audio, "audio")
        print(f"Sample text & audio records for dialect '{args.dialect}' successfully validated against schemas.")

if __name__ == "__main__":
    main()
