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
            "mms_1b": "facebook/mms-1b-all",
            "indicwhisper": "ai4bharat/indicwhisper-large-v3",
            "mt": "ai4bharat/indictrans2-indic-indic-1B",
            "indictrans2_3b": "ai4bharat/indictrans2-indic-indic-3B",
            "sarvam_llm": "sarvamai/sarvam-2b-v0.5",
            "airavata": "ai4bharat/airavata",
            "tts": "facebook/mms-tts-mwr",
            "indic_parler_tts": "ai4bharat/indic-parler-tts"
        },
        "dataset_path": "data/validated/mwr/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION", "PROVERB_RAG"]
    },
    "MTR": {
        "id": "MTR",
        "name": "Mewari",
        "native_name": "मेवाड़ी",
        "script": "Devanagari",
        "regions": ["Udaipur", "Chittorgarh", "Rajsamand", "Bhilwara"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-mtr",
            "mms_1b": "facebook/mms-1b-all",
            "indicwhisper": "ai4bharat/indicwhisper-large-v3",
            "mt": "ai4bharat/indictrans2-indic-indic-1B",
            "indictrans2_3b": "ai4bharat/indictrans2-indic-indic-3B",
            "sarvam_llm": "sarvamai/sarvam-2b-v0.5",
            "airavata": "ai4bharat/airavata",
            "tts": "facebook/mms-tts-mtr",
            "indic_parler_tts": "ai4bharat/indic-parler-tts"
        },
        "dataset_path": "data/validated/mtr/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION", "PROVERB_RAG"]
    },
    "DHD": {
        "id": "DHD",
        "name": "Dhundhari",
        "native_name": "ढूंढाड़ी",
        "script": "Devanagari",
        "regions": ["Jaipur", "Tonk", "Dausa"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-dhd",
            "mms_1b": "facebook/mms-1b-all",
            "indicwhisper": "ai4bharat/indicwhisper-large-v3",
            "mt": "ai4bharat/indictrans2-indic-indic-1B",
            "indictrans2_3b": "ai4bharat/indictrans2-indic-indic-3B",
            "sarvam_llm": "sarvamai/sarvam-2b-v0.5",
            "airavata": "ai4bharat/airavata",
            "tts": "facebook/mms-tts-dhd",
            "indic_parler_tts": "ai4bharat/indic-parler-tts"
        },
        "dataset_path": "data/validated/dhd/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION", "PROVERB_RAG"]
    },
    "HDT": {
        "id": "HDT",
        "name": "Hadoti",
        "native_name": "हाड़ौती",
        "script": "Devanagari",
        "regions": ["Kota", "Bundi", "Baran", "Jhalawar"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-hdt",
            "mms_1b": "facebook/mms-1b-all",
            "indicwhisper": "ai4bharat/indicwhisper-large-v3",
            "mt": "ai4bharat/indictrans2-indic-indic-1B",
            "indictrans2_3b": "ai4bharat/indictrans2-indic-indic-3B",
            "sarvam_llm": "sarvamai/sarvam-2b-v0.5",
            "airavata": "ai4bharat/airavata",
            "tts": "facebook/mms-tts-hdt",
            "indic_parler_tts": "ai4bharat/indic-parler-tts"
        },
        "dataset_path": "data/validated/hdt/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION", "PROVERB_RAG"]
    },
    "MWT": {
        "id": "MWT",
        "name": "Mewati",
        "native_name": "मेवाती",
        "script": "Devanagari",
        "regions": ["Alwar", "Bharatpur"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-mwt",
            "mms_1b": "facebook/mms-1b-all",
            "indicwhisper": "ai4bharat/indicwhisper-large-v3",
            "mt": "ai4bharat/indictrans2-indic-indic-1B",
            "indictrans2_3b": "ai4bharat/indictrans2-indic-indic-3B",
            "sarvam_llm": "sarvamai/sarvam-2b-v0.5",
            "airavata": "ai4bharat/airavata",
            "tts": "facebook/mms-tts-mwt",
            "indic_parler_tts": "ai4bharat/indic-parler-tts"
        },
        "dataset_path": "data/validated/mwt/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION", "PROVERB_RAG"]
    },
    "BGR": {
        "id": "BGR",
        "name": "Bagri",
        "native_name": "बागड़ी",
        "script": "Devanagari",
        "regions": ["Ganganagar", "Hanumangarh", "Churu"],
        "default_models": {
            "asr": "openai/whisper-large-v3-turbo-lora-bgr",
            "mms_1b": "facebook/mms-1b-all",
            "indicwhisper": "ai4bharat/indicwhisper-large-v3",
            "mt": "ai4bharat/indictrans2-indic-indic-1B",
            "indictrans2_3b": "ai4bharat/indictrans2-indic-indic-3B",
            "sarvam_llm": "sarvamai/sarvam-2b-v0.5",
            "airavata": "ai4bharat/airavata",
            "tts": "facebook/mms-tts-bgr",
            "indic_parler_tts": "ai4bharat/indic-parler-tts"
        },
        "dataset_path": "data/validated/bgr/",
        "supported_operations": ["ASR", "MT", "TTS", "DIALECT_ID", "NORMALIZATION", "PROVERB_RAG"]
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
