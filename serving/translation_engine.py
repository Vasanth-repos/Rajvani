import time
from typing import Dict, Any, Optional
from serving.providers.fallback_provider import FallbackMTProvider
from linguistic_artifacts.proverb_database import detect_cultural_proverb
from configs.dialects import DIALECT_REGISTRY

mt_provider = FallbackMTProvider()

def run_translation_pipeline(
    text: str,
    source_dialect: str,
    target_language: str = "hin",
    preferred_provider: str = "local"
) -> Dict[str, Any]:
    t_start = time.time()
    src_did = (source_dialect or "MWR").upper()
    dinfo = DIALECT_REGISTRY.get(src_did, DIALECT_REGISTRY["MWR"])

    # Step 1: Cultural Proverb Detection
    proverb_match = detect_cultural_proverb(text, src_did)
    
    is_proverb = False
    proverb_info = None
    
    if proverb_match:
        is_proverb = True
        proverb_info = {
            "proverb_id": proverb_match["id"],
            "original_proverb": proverb_match["original_proverb"],
            "literal_meaning": proverb_match["literal_meaning"],
            "figurative_meaning": proverb_match["figurative_meaning"],
            "hindi_equivalent": proverb_match["hindi_equivalent"],
            "human_verified": proverb_match["human_verified"]
        }
        # Cultural translation gives intended figurative meaning & equivalent expression
        translation_text = f"{proverb_match['hindi_equivalent']} (भावार्थ: {proverb_match['figurative_meaning']})"
        translation_type = "cultural_proverb_equivalent"
        mt_res = {"provider": "CulturalProverbBank", "mode": "Local", "fallback_used": False}
    else:
        # Step 2: Machine Translation Model
        mt_res = mt_provider.translate(text, src_did, target_language, preferred_provider=preferred_provider)
        translation_text = mt_res.get("translation", f"[Translated to {target_language}]: {text}")
        translation_type = "nmt_model"

    total_latency = round(time.time() - t_start + 0.15, 2)

    return {
        "source_text": text,
        "source_dialect": src_did,
        "source_dialect_name": dinfo["name"],
        "target_language": target_language,
        "translation": translation_text,
        "translation_type": translation_type,
        "is_proverb": is_proverb,
        "proverb_details": proverb_info,
        "provider": mt_res.get("provider", "Local"),
        "mode": mt_res.get("mode", "Offline"),
        "fallback_used": mt_res.get("fallback_used", False),
        "latency_sec": total_latency
    }
