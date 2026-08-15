import json
try:
    import pytest
except ImportError:
    pytest = None

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.schema.validate import validate_record, create_sample_text_record, create_sample_audio_record
from data.normalize_orthography import normalize_text
from data.splits.assign_split import assign_record_split, verify_file_path_read_access, materialize_splits
from scripts.generate_consent_artifacts import check_validated_consent_bases
from augmentation.back_translate import back_translate_batch
from augmentation.tts_bootstrap import bootstrap_tts_audio
from augmentation.audio_perturb import perturb_audio_records

def test_section2_schemas():
    sample_text = create_sample_text_record("mwr")
    sample_audio = create_sample_audio_record("mwr")
    assert validate_record(sample_text, "text") is True
    assert validate_record(sample_audio, "audio") is True

def test_section2_orthography_three_variant_collapse():
    # Synthetic test with 3 known variant spellings of one word collapsing to canonical form
    variant1 = "महारो नाम राम है।"
    variant2 = "म्हारो नाम राम है।"
    variant3 = "महोरा नाम राम है।"

    norm1, _ = normalize_text(variant1, "mwr")
    norm2, _ = normalize_text(variant2, "mwr")
    norm3, _ = normalize_text(variant3, "mwr")

    # All 3 collapse to the same canonical form in text_dialect
    assert norm1 == norm2 == norm3 == "म्हारो नाम राम है।"

    # Raw spellings remain distinct
    assert variant1 != variant2 != variant3

def test_section2_split_assignment_idempotence_and_cap():
    rec = {"id": "rec_fixed_01", "speaker_id": "spk_100", "source": "field_collection"}
    
    res1 = assign_record_split(dict(rec))
    res2 = assign_record_split(dict(rec))

    # Split assignment is deterministic and identical
    assert res1["split"] == res2["split"]
    assert res1["dev_subsplit"] == res2["dev_subsplit"]

    # Test dev cap (300 cap): Past cap, new records default to train
    heavy_counts = {"train": 1000, "dev": 300, "test": 500, "dev_promotion": 210, "dev_canary": 90}
    new_rec = {"id": "rec_new_999", "speaker_id": "spk_new_999", "source": "field_collection"}
    res_capped = assign_record_split(dict(new_rec), existing_counts=heavy_counts)
    
    # Dev cap reached -> forces split: train
    assert res_capped["split"] == "train"

def test_section2_all_augmentation_scripts_split_read_guard():
    test_path = ROOT_DIR / "data" / "splits" / "mwr" / "test.jsonl"
    canary_path = ROOT_DIR / "data" / "splits" / "mwr" / "dev_canary.jsonl"

    for script_fn, fn_args in [
        (back_translate_batch, ("mwr", str(test_path))),
        (bootstrap_tts_audio, ("mwr", str(test_path))),
        (perturb_audio_records, ("mwr", str(test_path)))
    ]:
        caught = False
        try:
            script_fn(*fn_args)
        except PermissionError:
            caught = True
        assert caught is True

    for script_fn, fn_args in [
        (back_translate_batch, ("mwr", str(canary_path))),
        (bootstrap_tts_audio, ("mwr", str(canary_path))),
        (perturb_audio_records, ("mwr", str(canary_path)))
    ]:
        caught = False
        try:
            script_fn(*fn_args)
        except PermissionError:
            caught = True
        assert caught is True

def test_section2_consent_audit():
    check_validated_consent_bases()

if __name__ == "__main__":
    test_section2_schemas()
    test_section2_orthography_three_variant_collapse()
    test_section2_split_assignment_idempotence_and_cap()
    test_section2_all_augmentation_scripts_split_read_guard()
    test_section2_consent_audit()
    print("test_section2: PASS")
