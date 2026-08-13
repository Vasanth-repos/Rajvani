import pytest
from pathlib import Path
import sys
import shutil

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.promote_checkpoint import evaluate_and_promote
from training.train_tts import run_tts_training

def test_section6_sequential_checkpoint_promotion_rejection():
    # Clean test checkpoint dir if exists
    test_dir = ROOT_DIR / "checkpoints" / "asr" / "mwr_test6_1"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # 1st checkpoint promotes (first_checkpoint)
    promoted1, meta1 = evaluate_and_promote("asr", "mwr_test6_1", "run_good_01", metric_name="wer")
    assert promoted1 is True
    assert meta1["reason"] == "first_checkpoint"

    # 2nd checkpoint synthetically made worse rejects (regression_detected)
    promoted2, meta2 = evaluate_and_promote("asr", "mwr_test6_1", "run_worse_poor_02", metric_name="wer")
    assert promoted2 is False
    assert meta2["reason"] == "regression_detected"

def test_section6_metric_direction_awareness():
    test_dir = ROOT_DIR / "checkpoints" / "tts" / "mwr_test6_2"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # MCD objective metric configured as lower_is_better (first_checkpoint promotes)
    promoted1, _ = evaluate_and_promote("tts", "mwr_test6_2", "run_mcd_good_01", metric_name="mcd")
    assert promoted1 is True

    # Higher score for lower_is_better metric must be rejected
    promoted2, meta2 = evaluate_and_promote("tts", "mwr_test6_2", "run_mcd_worse_poor_02", metric_name="mcd")
    assert promoted2 is False
    assert meta2["metric_direction"] == "lower_is_better"

def test_section6_tts_voice_clone_consent_gating():
    # If eligible voice_clone_ok audio count < 20, refuses to train and returns INSUFFICIENT_DATA
    res = run_tts_training("mwt_unfunded", backend="mms")
    assert res == "INSUFFICIENT_DATA"
