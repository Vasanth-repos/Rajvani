import logging
from typing import Dict, Any, Optional
from serving.providers.bhashini_provider import BhashiniASRProvider, BhashiniMTProvider, BhashiniTTSProvider
from serving.providers.local_provider import LocalASRProvider, LocalMTProvider, LocalTTSProvider

logger = logging.getLogger(__name__)

class FallbackASRProvider:
    def __init__(self):
        self.bhashini = BhashiniASRProvider()
        self.local = LocalASRProvider()

    def transcribe(self, audio_path: str, dialect_id: Optional[str] = None, preferred_provider: str = "local") -> Dict[str, Any]:
        if preferred_provider.lower() == "bhashini":
            try:
                res = self.bhashini.transcribe(audio_path, dialect_id)
                res["fallback_used"] = False
                return res
            except Exception as e:
                logger.warning(f"Bhashini ASR unavailable: {e}. Falling back to Local model.")
        
        res = self.local.transcribe(audio_path, dialect_id)
        res["fallback_used"] = (preferred_provider.lower() == "bhashini")
        if res["fallback_used"]:
            res["notice"] = "Bhashini unavailable. Attempting local provider..."
        return res

class FallbackMTProvider:
    def __init__(self):
        self.bhashini = BhashiniMTProvider()
        self.local = LocalMTProvider()

    def translate(self, text: str, source_dialect: str, target_lang: str = "hin", preferred_provider: str = "local") -> Dict[str, Any]:
        if preferred_provider.lower() == "bhashini":
            try:
                res = self.bhashini.translate(text, source_dialect, target_lang)
                res["fallback_used"] = False
                return res
            except Exception as e:
                logger.warning(f"Bhashini MT unavailable: {e}. Falling back to Local model.")
        
        res = self.local.translate(text, source_dialect, target_lang)
        res["fallback_used"] = (preferred_provider.lower() == "bhashini")
        if res["fallback_used"]:
            res["notice"] = "Bhashini unavailable. Attempting local provider..."
        return res

class FallbackTTSProvider:
    def __init__(self):
        self.bhashini = BhashiniTTSProvider()
        self.local = LocalTTSProvider()

    def synthesize(self, text: str, dialect_id: str, backend: str = "mms", preferred_provider: str = "local") -> Dict[str, Any]:
        if preferred_provider.lower() == "bhashini":
            try:
                res = self.bhashini.synthesize(text, dialect_id, backend)
                res["fallback_used"] = False
                return res
            except Exception as e:
                logger.warning(f"Bhashini TTS unavailable: {e}. Falling back to Local model.")
        
        res = self.local.synthesize(text, dialect_id, backend)
        res["fallback_used"] = (preferred_provider.lower() == "bhashini")
        if res["fallback_used"]:
            res["notice"] = "Bhashini unavailable. Attempting local provider..."
        return res
