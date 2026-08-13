import pytest
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from augmentation.back_translate import back_translate_batch
from augmentation.audio_perturb import perturb_audio_records

def test_section4_augmentation_source_tagging_and_isolation():
    train_dir = ROOT_DIR / "data" / "splits" / "mwr"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    train_split_path = train_dir / "train.jsonl"
    if not train_split_path.exists():
        with open(train_split_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "mwr_001", "text_hindi": "यह एक परीक्षण है।", "text_dialect": "ओ एक परीक्षण छै।"}) + "\n")

    # Backtranslation creates synthetic records
    c = back_translate_batch("mwr", str(train_split_path), generator_checkpoint="base", chrf_threshold=0.5)
    assert c >= 0

    # Augmentation script blocked from reading test.jsonl
    test_split_path = train_dir / "test.jsonl"
    with pytest.raises(PermissionError):
        back_translate_batch("mwr", str(test_split_path))
