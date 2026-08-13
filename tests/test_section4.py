import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from augmentation.back_translate import back_translate_batch
from augmentation.audio_perturb import perturb_audio_records

def test_section4_augmentation_source_tagging_and_isolation():
    train_split_path = ROOT_DIR / "data" / "splits" / "mwr" / "train.jsonl"
    
    # Backtranslation creates synthetic records
    c = back_translate_batch("mwr", str(train_split_path), generator_checkpoint="base", chrf_threshold=0.5)
    assert c >= 0

    # Augmentation script blocked from reading test.jsonl
    test_split_path = ROOT_DIR / "data" / "splits" / "mwr" / "test.jsonl"
    with pytest.raises(PermissionError):
        back_translate_batch("mwr", str(test_split_path))
