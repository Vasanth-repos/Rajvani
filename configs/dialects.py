from typing import Dict, Any, List

DIALECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "MWR": {
        "id": "MWR",
        "name": "Marwari",
        "native_name": "मारवाड़ी",
        "script": "Devanagari",
        "regions": ["Jodhpur", "Bikaner", "Barmer", "Jaisalmer", "Nagaur"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-mwr",
            "mt": "ai4bharat/indictrans2-1b-lora-mwr",
            "tts": "facebook/mms-tts-mwr"
        },
        "dataset_path": "data/validated/mwr/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION"]
    },
    "MTR": {
        "id": "MTR",
        "name": "Mewari",
        "native_name": "मेवाड़ी",
        "script": "Devanagari",
        "regions": ["Udaipur", "Chittorgarh", "Rajsamand", "Bhilwara"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-mtr",
            "mt": "ai4bharat/indictrans2-1b-lora-mtr",
            "tts": "facebook/mms-tts-mtr"
        },
        "dataset_path": "data/validated/mtr/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION"]
    },
    "DHD": {
        "id": "DHD",
        "name": "Dhundhari",
        "native_name": "ढूंढाड़ी",
        "script": "Devanagari",
        "regions": ["Jaipur", "Tonk", "Dausa"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-dhd",
            "mt": "ai4bharat/indictrans2-1b-lora-dhd",
            "tts": "facebook/mms-tts-dhd"
        },
        "dataset_path": "data/validated/dhd/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION"]
    },
    "HDT": {
        "id": "HDT",
        "name": "Hadoti",
        "native_name": "हाड़ौती",
        "script": "Devanagari",
        "regions": ["Kota", "Bundi", "Baran", "Jhalawar"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-hdt",
            "mt": "ai4bharat/indictrans2-1b-lora-hdt",
            "tts": "facebook/mms-tts-hdt"
        },
        "dataset_path": "data/validated/hdt/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION"]
    },
    "MWT": {
        "id": "MWT",
        "name": "Mewati",
        "native_name": "मेवाती",
        "script": "Devanagari",
        "regions": ["Alwar", "Bharatpur"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-mwt",
            "mt": "ai4bharat/indictrans2-1b-lora-mwt",
            "tts": "facebook/mms-tts-mwt"
        },
        "dataset_path": "data/validated/mwt/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION"]
    },
    "BGR": {
        "id": "BGR",
        "name": "Bagri",
        "native_name": "बागड़ी",
        "script": "Devanagari",
        "regions": ["Ganganagar", "Hanumangarh", "Churu"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-bgr",
            "mt": "ai4bharat/indictrans2-1b-lora-bgr",
            "tts": "facebook/mms-tts-bgr"
        },
        "dataset_path": "data/validated/bgr/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION"]
    }
}

def get_dialect_info(dialect_id: str) -> Dict[str, Any]:
    """Retrieves metadata for a dialect ID (case-insensitive)."""
    key = (dialect_id or "").upper().split()[0]
    if key not in DIALECT_REGISTRY:
        raise ValueError(f"Unknown dialect ID '{dialect_id}'. Supported IDs: {list(DIALECT_REGISTRY.keys())}")
    return DIALECT_REGISTRY[key]

def list_dialects() -> List[Dict[str, Any]]:
    """Returns list of all registered dialect metadata objects."""
    return list(DIALECT_REGISTRY.values())

def validate_dialect_id(dialect_id: str) -> bool:
    """Checks if a dialect ID is registered."""
    key = (dialect_id or "").upper().split()[0]
    return key in DIALECT_REGISTRY
