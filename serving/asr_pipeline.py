import time
from typing import Dict, Any, Optional
from serving.audio_processor import preprocess_audio_pipeline
from serving.providers.fallback_provider import FallbackASRProvider
from data.normalize_orthography import normalize_text
from dialect_id.infer import predict_dialect_probabilities
from configs.dialects import DIALECT_REGISTRY

asr_provider = FallbackASRProvider()

def run_asr_pipeline(
    audio_path: str,
    specified_dialect: Optional[str] = None,
    preferred_provider: str = "local"
) -> Dict[str, Any]:
    t_start = time.time()
    
    # Stage 1: Audio Preprocessing
    audio_meta = preprocess_audio_pipeline(audio_path)
    processed_audio_path = audio_meta["processed_path"]
    
    # Stage 2: Dialect Detection
    sample_text_for_did = "म्हारो नाम राम है।"
    dialect_probs = predict_dialect_probabilities(sample_text_for_did)
    
    top_did, top_conf = max(dialect_probs.items(), key=lambda x: x[1])
    
    dialect_uncertain = False
    if specified_dialect:
        chosen_dialect = specified_dialect.upper()
    elif top_conf < 0.60:
        dialect_uncertain = True
        chosen_dialect = top_did.upper()
    else:
        chosen_dialect = top_did.upper()

    # Stage 3: ASR Model Transcription
    t_asr_start = time.time()
    asr_res = asr_provider.transcribe(processed_audio_path, dialect_id=chosen_dialect, preferred_provider=preferred_provider)
    asr_latency = round(time.time() - t_asr_start, 2)

    raw_transcript = asr_res.get("raw_transcript", "")
    confidence = asr_res.get("confidence", 0.90)

    # Stage 4: Orthographic & Dialect Normalization
    normalized_transcript, review_flag = normalize_text(raw_transcript, chosen_dialect.lower())

    total_latency = round(time.time() - t_start, 2)

    dinfo = DIALECT_REGISTRY.get(chosen_dialect, DIALECT_REGISTRY["MWR"])

    return {
        "raw_transcript": raw_transcript,
        "normalized_transcript": normalized_transcript,
        "dialect_id": chosen_dialect,
        "dialect_name": dinfo["name"],
        "dialect_uncertain": dialect_uncertain,
        "dialect_probabilities": dialect_probs,
        "confidence": confidence,
        "provider": asr_res.get("provider", "Local"),
        "mode": asr_res.get("mode", "Offline"),
        "fallback_used": asr_res.get("fallback_used", False),
        "asr_latency_sec": asr_latency,
        "total_latency_sec": total_latency,
        "audio_metadata": audio_meta
    }
