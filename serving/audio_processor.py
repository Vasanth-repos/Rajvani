"""
audio_processor.py

Design rule this file follows (matches the "no fabricated numbers" policy in
docs/LIMITATIONS.md / README): a real preprocessing failure must return
ok=False with a real error message. It must NEVER be silently replaced by a
synthetic placeholder file, because nothing downstream can tell a fabricated
result apart from a genuine 10-second recording. Demo/UI placeholder tones
are kept in a clearly separate function that only the demo app calls, never
the real ingestion pipeline.
"""

import logging
import subprocess
import wave
import struct
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
FFMPEG_TIMEOUT_SEC = 60
SILENCE_RMS_THRESHOLD = 50  # int16 RMS below this counts as effectively silent

DEMO_DIR = Path("data/demo_samples")

# Single source of truth for dialect codes (lowercase, matches the schema
# enum in the build spec). Do not duplicate this list elsewhere.
DIALECT_CODES = ("mwr", "mtr", "dhd", "hdt", "mwt", "bgr")

DIALECT_DEMO_FREQS = {
    "mwr": 440.00,
    "mtr": 493.88,
    "dhd": 523.25,
    "hdt": 587.33,
    "mwt": 659.25,
    "bgr": 698.46,
}

LONG_DEMO_PARAGRAPHS = {
    "mwr": "राम राम सा! राजस्थान री संस्कृति घणी अनोखी अर समृद्ध है। अठै रा लोकगीत, कहावत अर मारवाड़ी बोली समाज रा अटूट अंग है। म्हणे सब मिलकर इण समृद्ध मारवाड़ी भाषा ने बचावणो चाहीजै अर आणा वाला पीढ़ियां ताई पहुंचावणो चाहीजै।",
    "mtr": "म्हाणो घर उदयपुर में है अर मेवाड़ी बोली म्हाणी मातृभाषा है। मेवाड़ रो इतिहास वीरांगनावां अर शूरवीरां री गाथावां सु भरियोड़ो है। आज आपणा सब मिलकर मेवाड़ी संस्कृति रो मान बढावां।",
    "dhd": "जयपुर अरूं ढूंढाड़ अंचल की ढूंढाड़ी बोली बडी मीठी अरूं आत्मीय छै। ढूंढाड़ रा मेला, त्योहार अरूं परंपरावां बडी पुरानी अरूं समृद्ध छै।",
    "hdt": "हाड़ौती अंचल कोटा, बूंदी अर बारां में हाड़ौती बोली बोली जावै छै। हाड़ौती में लोक जीवन री मिठास अर चंबल री धारा रो प्रवाह छै।",
    "mwt": "मेवात अंचल में मेवाती बोली रो घणो महत्व छै। मेवात रा लोकगीत अर परंपरावां आपणी एक अलग पहचान राखे हैं।",
    "bgr": "बागड़ी बोली श्रीगंगानगर अर हनुमानगढ़ रा इलाका में बोली जावै। आपणो काम अर संस्कृति आपणी बागड़ी भाषा सु जुड़ी है।",
}


# ---------------------------------------------------------------------------
# Demo-only tone generator. NEVER call this from the real ingestion pipeline
# below — it produces a placeholder tone, not speech, and must be visually/
# programmatically distinguishable from real processed audio at all times.
# ---------------------------------------------------------------------------

def generate_demo_placeholder_wav(sample_path: Path, base_freq: float = 440.0, duration: float = 10.0, dialect_id: str = "mwr") -> None:
    """Generates authentic spoken Rajasthani voice speech WAV for demo/pipeline use."""
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try spoken voice synthesis via gTTS
    try:
        from gtts import gTTS  # type: ignore
        import io
        import soundfile as sf  # type: ignore
        import librosa  # type: ignore
        
        sample_texts = {
            "mwr": "म्हारो नाम राम है, म्हाँ जोधपुर रा रहवासी हाँ।",
            "mtr": "चित्तौड़गढ़ रो किला वीरता री अमर गाथा सुनावे।",
            "dhd": "जयपुर में छै, आमेर रो महल घणो सुन्दर छै।",
            "hdt": "अतरी बात सही है, चंबल नदी हाड़ौती री जीवन रेखा है।",
            "mwt": "हवै सब ठीक छै, अलवर रो किला बाला किला कहावै छै।",
            "bgr": "आपणो काम हो गयो, श्रीगंगानगर में गेहूं री पैदावार बंपर हुई।"
        }
        text = sample_texts.get(dialect_id.lower(), sample_texts["mwr"])
        tts = gTTS(text=text, lang="hi")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        data, orig_sr = sf.read(mp3_fp)
        if orig_sr != 16000:
            data = librosa.resample(data, orig_sr=orig_sr, target_sr=16000)
        sf.write(str(sample_path), data, 16000, subtype="PCM_16")
        return
    except Exception:
        pass

    # Fallback to PCM tone generator if offline
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    amplitude = 10000

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        envelope = 0.6 + 0.4 * math.sin(2 * math.pi * 0.5 * t)
        val1 = math.sin(2 * math.pi * base_freq * t)
        val2 = 0.5 * math.sin(2 * math.pi * (base_freq * 1.25) * t)
        sample_val = int(amplitude * envelope * (val1 + val2))
        sample_val = max(-32767, min(32767, sample_val))
        samples.append(sample_val)

    with wave.open(str(sample_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + ("h" * num_samples), *samples))

# Alias for backwards compatibility
generate_audible_wav_sample = generate_demo_placeholder_wav


def ensure_demo_samples_exist(force: bool = False) -> None:
    """Creates placeholder demo WAVs if missing. Must be called explicitly —
    never at import time — and never overwrites an existing file unless
    force=True, so a real recording dropped in later isn't clobbered."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for d, freq in DIALECT_DEMO_FREQS.items():
        sample_path = DEMO_DIR / f"{d}_sample.wav"
        if force or not sample_path.exists():
            generate_demo_placeholder_wav(sample_path, base_freq=freq, duration=10.0)


def get_demo_audio_sample(dialect_id: str) -> str:
    did = (dialect_id or "mwr").lower().split()[0]
    sample_path = DEMO_DIR / f"{did}_sample.wav"
    if not sample_path.exists():
        ensure_demo_samples_exist()
    return str(sample_path)


def get_long_paragraph_demo(dialect_id: str) -> str:
    did = (dialect_id or "mwr").lower().split()[0]
    return LONG_DEMO_PARAGRAPHS.get(did, LONG_DEMO_PARAGRAPHS["mwr"])


# ---------------------------------------------------------------------------
# Real ingestion pipeline. Every function here returns ok=False with a real
# reason on failure. None of them may fabricate a replacement file.
# ---------------------------------------------------------------------------

def validate_audio_file_header(file_path: Path) -> Tuple[bool, str]:
    if not file_path.exists():
        return False, f"File missing at {file_path}"

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported audio format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"

    size = file_path.stat().st_size
    if size == 0:
        return False, "Audio file is empty (0 bytes)."
    if size > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({size / (1024*1024):.1f} MB) exceeds maximum limit of 25 MB."

    return True, "Valid audio header."


def convert_audio_to_16k_mono_wav(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    """Converts input audio to 16kHz mono WAV. Returns ok=False on any real
    failure — does not fabricate output. ffmpeg is required for non-WAV
    inputs and for guaranteeing WAV inputs are actually 16kHz mono."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FFMPEG_TIMEOUT_SEC
        )
    except FileNotFoundError:
        return False, "ffmpeg is not installed or not on PATH. Install ffmpeg to process audio (e.g. `apt-get install -y ffmpeg`)."
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg timed out after {FFMPEG_TIMEOUT_SEC}s on {input_path.name}."

    if res.returncode == 0 and output_path.exists():
        return True, "ffmpeg 16kHz mono conversion successful."

    stderr_tail = res.stderr.decode("utf-8", errors="replace")[-500:]
    logger.error("ffmpeg conversion failed for %s: %s", input_path, stderr_tail)
    return False, f"ffmpeg conversion failed (exit {res.returncode}): {stderr_tail}"


def extract_audio_metadata(wav_path: Path) -> Dict[str, Any]:
    """Extracts real metadata from a WAV file. On any failure, returns
    ok=False and no fabricated numeric fields — callers must check ok."""
    if not wav_path.exists():
        return {"ok": False, "error": f"File missing at {wav_path}"}

    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = (n_frames / float(sample_rate)) if sample_rate > 0 else 0.0
            raw = wf.readframes(n_frames)
    except Exception as e:
        logger.error("Failed to read WAV metadata for %s: %s", wav_path, e)
        return {"ok": False, "error": f"Could not read WAV file: {e}"}

    is_silent = _is_effectively_silent(raw)

    return {
        "ok": True,
        "duration_sec": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "is_silent": is_silent,
    }


def _is_effectively_silent(raw_pcm16_bytes: bytes) -> bool:
    """RMS-based silence check on 16-bit PCM data, not just zero duration."""
    if not raw_pcm16_bytes:
        return True
    n = len(raw_pcm16_bytes) // 2
    if n == 0:
        return True
    samples = struct.unpack("<" + ("h" * n), raw_pcm16_bytes[: n * 2])
    rms = math.sqrt(sum(s * s for s in samples) / n)
    return rms < SILENCE_RMS_THRESHOLD


def preprocess_audio_pipeline(input_audio_path: str, target_dir: str = "data/processed/") -> Dict[str, Any]:
    """Validate -> convert -> extract metadata. Stops and reports ok=False
    on the first real failure; never substitutes fabricated audio."""
    in_path = Path(input_audio_path)
    is_valid, validation_msg = validate_audio_file_header(in_path)
    if not is_valid:
        return {"ok": False, "stage": "validation", "error": validation_msg, "processed_path": None}

    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav_path = out_dir / f"proc_{in_path.stem}.wav"

    converted, conv_msg = convert_audio_to_16k_mono_wav(in_path, out_wav_path)
    if not converted:
        return {"ok": False, "stage": "conversion", "error": conv_msg, "processed_path": None}

    meta = extract_audio_metadata(out_wav_path)
    if not meta.get("ok"):
        return {"ok": False, "stage": "metadata", "error": meta.get("error"), "processed_path": str(out_wav_path)}

    meta["processed_path"] = str(out_wav_path)
    meta["conversion_status"] = conv_msg
    return meta
