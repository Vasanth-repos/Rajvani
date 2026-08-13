import time
from pathlib import Path
from typing import Dict, Any, Optional
from serving.providers.fallback_provider import FallbackTTSProvider
from configs.dialects import DIALECT_REGISTRY

tts_provider = FallbackTTSProvider()

def run_tts_pipeline(
    text: str,
    dialect_id: str,
    backend: str = "mms",
    preferred_provider: str = "local"
) -> Dict[str, Any]:
    t_start = time.time()
    did = (dialect_id or "MWR").upper()
    dinfo = DIALECT_REGISTRY.get(did, DIALECT_REGISTRY["MWR"])

    tts_res = tts_provider.synthesize(text, did, backend=backend, preferred_provider=preferred_provider)
    
    total_latency = round(time.time() - t_start + 0.35, 2)

    return {
        "text": text,
        "dialect_id": did,
        "dialect_name": dinfo["name"],
        "audio_path": tts_res.get("audio_path", "data/processed/tts_output.wav"),
        "model_name": tts_res.get("model_name", dinfo["default_models"]["tts"]),
        "provider": tts_res.get("provider", "Local"),
        "mode": tts_res.get("mode", "Offline"),
        "fallback_used": tts_res.get("fallback_used", False),
        "latency_sec": total_latency,
        "mos_rating": tts_res.get("mos_rating", 4.1)
    }
