import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from configs.dialects import list_dialects, get_dialect_info, validate_dialect_id, DIALECT_REGISTRY
from serving.audio_processor import preprocess_audio_pipeline
from serving.providers.fallback_provider import FallbackASRProvider, FallbackMTProvider, FallbackTTSProvider
from serving.providers.status import get_provider_status
from linguistic_artifacts.proverb_database import list_proverbs, search_proverbs, detect_cultural_proverb
from eval.asr_eval import compute_wer, compute_cer
from eval.mt_eval import compute_bleu_score, compute_chrf_score
from eval.tts_eval import calculate_mean_mos
from eval.human_feedback import record_user_feedback, get_feedback_summary

def test_centralized_dialect_registry():
    assert len(DIALECT_REGISTRY) == 6
    for did in ["MWR", "MTR", "DHD", "HDT", "MWT", "BGR"]:
        assert validate_dialect_id(did) is True
        info = get_dialect_info(did)
        assert info["id"] == did
        assert "default_models" in info
        assert "supported_operations" in info

def test_audio_preprocessing_pipeline(tmp_path):
    dummy_wav = tmp_path / "sample.wav"
    dummy_wav.write_bytes(b"RIFF44WAVEfmt ")
    meta = preprocess_audio_pipeline(str(dummy_wav), target_dir=str(tmp_path / "proc"))
    assert "processed_path" in meta
    assert meta["sample_rate"] == 16000

def test_provider_fallback_architecture():
    asr_p = FallbackASRProvider()
    res = asr_p.transcribe("data/processed/sample.wav", dialect_id="MWR", preferred_provider="bhashini")
    assert res["provider"] == "Local"
    assert res["mode"] == "Offline"
    assert res["fallback_used"] is True

def test_provider_status_panel():
    st = get_provider_status()
    assert "providers" in st
    assert st["mode"] == "Offline"

def test_proverb_database_and_cultural_matching():
    proverbs = list_proverbs()
    assert len(proverbs) >= 6
    match = detect_cultural_proverb("अेक साधे सब सधै", "MWR")
    assert match is not None
    assert match["id"] == "mwr_prv_001"
    assert match["human_verified"] is True

def test_metrics_evaluation_functions():
    wer = compute_wer(["म्हारो नाम राम है"], ["म्हारो नाम श्याम है"])
    assert wer > 0.0
    cer = compute_cer(["म्हारो"], ["म्हारो"])
    assert cer == 0.0
    bleu = compute_bleu_score("म्हारो नाम राम है", "म्हारो नाम राम है")
    assert bleu == 100.0
    chrf = compute_chrf_score("म्हारो", "म्हारो")
    assert chrf == 100.0

def test_human_feedback_recorder():
    rec = record_user_feedback(5, 5, 5, 5, 5, comments="Excellent test feedback", dialect_id="MWR")
    assert rec["overall_usefulness"] == 5
    summary = get_feedback_summary()
    assert summary["total_trials"] >= 1
