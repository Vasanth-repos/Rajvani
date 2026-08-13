import os
from typing import Dict, Any
from serving.providers.bhashini_provider import BhashiniASRProvider

def get_provider_status() -> Dict[str, Any]:
    bhashini_active = bool(os.getenv("BHASHINI_API_KEY"))
    return {
        "providers": {
            "bhashini": {
                "name": "Bhashini API",
                "status": "Online" if bhashini_active else "Offline",
                "badge": "● Online" if bhashini_active else "● Offline"
            },
            "local_asr": {
                "name": "Local ASR Model",
                "status": "Ready",
                "badge": "● Ready"
            },
            "local_mt": {
                "name": "Local MT Model",
                "status": "Ready",
                "badge": "● Ready"
            },
            "local_tts": {
                "name": "Local TTS Model",
                "status": "Ready",
                "badge": "● Ready"
            }
        },
        "mode": "Online" if bhashini_active else "Offline"
    }
