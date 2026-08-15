"""
serving/neural_mt.py

Neural Machine Translation inference module using Meta NLLB-200 (facebook/nllb-200-distilled-600M).
Supports translation from Rajasthani dialects to Standard Hindi (hin_Deva) and English (eng_Latn).
"""

import logging
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

_NLLB_MODEL = None
_NLLB_TOKENIZER = None

MODEL_ID = "facebook/nllb-200-distilled-600M"

LANG_CODE_MAP = {
    "hin": "hin_Deva",
    "hi": "hin_Deva",
    "eng": "eng_Latn",
    "en": "eng_Latn"
}

def get_nllb_pipeline():
    global _NLLB_MODEL, _NLLB_TOKENIZER
    if _NLLB_MODEL is None or _NLLB_TOKENIZER is None:
        logger.info(f"Loading neural NMT model: {MODEL_ID}...")
        _NLLB_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        _NLLB_MODEL.eval()
    return _NLLB_TOKENIZER, _NLLB_MODEL

def translate_neural(text: str, target_lang: str = "hin", max_length: int = 128) -> str:
    """Translates input dialect text to target language using neural seq2seq NLLB-200."""
    if not text or not text.strip():
        return ""
    
    clean_text = text.strip()
    try:
        tokenizer, model = get_nllb_pipeline()
        tgt_code = LANG_CODE_MAP.get(target_lang.lower(), "hin_Deva")
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

        inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=2
            )
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return output_text if output_text else clean_text
    except Exception as e:
        logger.warning(f"Neural MT inference fallback triggered: {e}")
        return clean_text
