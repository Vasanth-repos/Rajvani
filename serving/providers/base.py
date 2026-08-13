from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseASRProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, dialect_id: Optional[str] = None) -> Dict[str, Any]:
        """Transcribes audio file to dialect text."""
        pass

class BaseMTProvider(ABC):
    @abstractmethod
    def translate(self, text: str, source_dialect: str, target_lang: str = "hin") -> Dict[str, Any]:
        """Translates text from source dialect to target language."""
        pass

class BaseTTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, dialect_id: str, backend: str = "mms") -> Dict[str, Any]:
        """Synthesizes text to dialect audio."""
        pass
