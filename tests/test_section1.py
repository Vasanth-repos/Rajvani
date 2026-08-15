try:
    import pytest
except ImportError:
    pytest = None

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from configs.dialects import list_dialects, get_dialect_info, validate_dialect_id, DIALECT_REGISTRY
from serving.audio_processor import preprocess_audio_pipeline, get_demo_audio_sample
from serving.providers.fallback_provider import FallbackASRProvider, FallbackMTProvider, FallbackTTSProvider
from serving.providers.status import get_provider_status
from linguistic_artifacts.proverb_database import list_proverbs, search_proverbs, detect_cultural_proverb
from eval.asr_eval import compute_wer, compute_cer, get_baseline_vs_finetuned_comparison
from eval.mt_eval import compute_bleu_score, compute_chrf_score
from eval.tts_eval import calculate_mean_mos
from eval.human_feedback import record_user_feedback, get_feedback_summary
from eval.cross_dialect_transfer import get_cross_dialect_matrix, explain_na_cell
from active_learning.human_verifier import save_human_verified_transcript, get_verified_dataset_count

def test_centralized_dialect_registry():
    assert len(DIALECT_REGISTRY) == 6
    for did in ["MWR", "MTR", "DHD", "HDT", "MWT", "BGR"]:
        assert validate_dialect_id(did) is True
        info = get_dialect_info(did)
        assert info["id"] == did
        assert "default_models" in info

def test_demo_audio_samples():
    sample_path = get_demo_audio_sample("MWR")
    assert Path(sample_path).exists()
    assert sample_path.endswith(".wav")

def test_human_transcript_correction():
    res = save_human_verified_transcript("म्हारो नाम राम है", "म्हारो नाम राम है।", "MWR")
    assert res["status"] == "success"
    assert get_verified_dataset_count() > 100

def test_baseline_vs_finetuned_comparison():
    comp = get_baseline_vs_finetuned_comparison()
    assert len(comp) == 6
    assert "baseline_wer" in comp[0]
    assert "finetuned_wer" in comp[0]

def test_transfer_matrix_modes_and_na_explanation():
    zero_mat = get_cross_dialect_matrix("asr", mode="zero_shot")
    fine_mat = get_cross_dialect_matrix("asr", mode="finetuned")
    assert zero_mat["MTR"]["BGR"] == "N/A"
    assert fine_mat["MTR"]["BGR"] != "N/A"
    
    na_info = explain_na_cell("MTR", "BGR")
    assert "status" in na_info
    assert na_info["status"] == "Not Evaluated (N/A)"

def test_provider_fallback_architecture():
    asr_p = FallbackASRProvider()
    res = asr_p.transcribe("data/processed/sample.wav", dialect_id="MWR", preferred_provider="bhashini")
    assert res["provider"] == "Local"
    assert res["mode"] == "Offline"
    assert res["fallback_used"] is True

def test_proverb_database_and_featured_cards():
    proverbs = list_proverbs()
    assert len(proverbs) >= 6
    match = detect_cultural_proverb("अेक साधे सब सधै", "MWR")
    assert match is not None
    assert match["id"] == "mwr_prv_001"

if __name__ == "__main__":
    test_centralized_dialect_registry()
    test_demo_audio_samples()
    test_human_transcript_correction()
    test_baseline_vs_finetuned_comparison()
    test_transfer_matrix_modes_and_na_explanation()
    test_provider_fallback_architecture()
    test_proverb_database_and_featured_cards()
    print("test_section1: PASS")
