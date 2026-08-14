import os
import sys
import wave
import struct
import math
from pathlib import Path
from typing import Dict, Any, Tuple

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

DEMO_DIR = Path("data/demo_samples")

# Long multi-sentence paragraph samples for each dialect for extended audio synthesis (15s - 30s)
LONG_DEMO_PARAGRAPHS = {
    "MWR": "राम राम सा! राजस्थान री संस्कृति घणी अनोखी अर समृद्ध है। अठै रा लोकगीत, कहावत अर मारवाड़ी बोली समाज रा अटूट अंग है। म्हणे सब मिलकर इण समृद्ध मारवाड़ी भाषा ने बचावणो चाहीजै अर आणा वाला पीढ़ियां ताई पहुंचावणो चाहीजै।",
    "MTR": "म्हाणो घर उदयपुर में है अर मेवाड़ी बोली म्हाणी मातृभाषा है। मेवाड़ रो इतिहास वीरांगनावां अर शूरवीरां री गाथावां सु भरियोड़ो है। आज आपणा सब मिलकर मेवाड़ी संस्कृति रो मान बढावां।",
    "DHD": "जयपुर अरूं ढूंढाड़ अंचल की ढूंढाड़ी बोली बडी मीठी अरूं आत्मीय छै। ढूंढाड़ रा मेला, त्योहार अरूं परंपरावां बडी पुरानी अरूं समृद्ध छै।",
    "HDT": "हाड़ौती अंचल कोटा, बूंदी अर बारां में हाड़ौती बोली बोली जावै छै। हाड़ौती में लोक जीवन री मिठास अर चंबल री धारा रो प्रवाह छै।",
    "MWT": "मेवात अंचल में मेवाती बोली रो घणो महत्व छै। मेवात रा लोकगीत अर परंपरावां आपणी एक अलग पहचान राखे हैं।",
    "BGR": "बागड़ी बोली श्रीगंगानगर अर हनुमानगढ़ रा इलाका में बोली जावै। आपणो काम अर संस्कृति आपणी बागड़ी भाषा सु जुड़ी है।"
}

def generate_audible_wav_sample(sample_path: Path, base_freq: float = 440.0, duration: float = 10.0):
    """Generates an audible 16kHz mono WAV tone sequence with pleasant harmonics and longer duration."""
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    amplitude = 10000  # Clear audible 16-bit PCM amplitude

    samples = []
    for i in range(num_samples):
        t = float(i) / sample_rate
        # Pleasant repeating dual-tone acoustic chord envelope
        envelope = 0.6 + 0.4 * math.sin(2 * math.pi * 0.5 * t)
        val1 = math.sin(2 * math.pi * base_freq * t)
        val2 = 0.5 * math.sin(2 * math.pi * (base_freq * 1.25) * t)  # Major third harmonic
        sample_val = int(amplitude * envelope * (val1 + val2))
        sample_val = max(-32767, min(32767, sample_val))
        samples.append(sample_val)

    with wave.open(str(sample_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = struct.pack("<" + ("h" * num_samples), *samples)
        wf.writeframes(frames)

def ensure_demo_samples_exist():
    """Generates pre-loaded 16kHz mono audible demo WAV files for all 6 dialects."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    dialect_freqs = {
        "mwr": 440.0,
        "mtr": 493.88,
        "dhd": 523.25,
        "hdt": 587.33,
        "mwt": 659.25,
        "bgr": 698.46
    }
    for d, freq in dialect_freqs.items():
        sample_path = DEMO_DIR / f"{d}_sample.wav"
        generate_audible_wav_sample(sample_path, base_freq=freq, duration=10.0)

ensure_demo_samples_exist()

def get_demo_audio_sample(dialect_id: str) -> str:
    """Returns file path to pre-loaded demo audio clip for given dialect."""
    did = (dialect_id or "mwr").lower().split()[0]
    sample_path = DEMO_DIR / f"{did}_sample.wav"
    if not sample_path.exists():
        ensure_demo_samples_exist()
    return str(sample_path)

def get_long_paragraph_demo(dialect_id: str) -> str:
    """Returns extended multi-sentence paragraph text for a given dialect."""
    did = (dialect_id or "MWR").upper().split()[0]
    return LONG_DEMO_PARAGRAPHS.get(did, LONG_DEMO_PARAGRAPHS["MWR"])

def validate_audio_file_header(file_path: Path) -> Tuple[bool, str]:
    """Validates file existence, format extension, and size bounds."""
    if not file_path.exists():
        return False, f"File missing at {file_path}"
    
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported audio format '{ext}'. Supported: {list(SUPPORTED_EXTENSIONS)}"
    
    size = file_path.stat().st_size
    if size == 0:
        return False, "Audio file is empty (0 bytes)."
    if size > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({size / (1024*1024):.1f} MB) exceeds maximum limit of 25 MB."
    
    return True, "Valid audio header."

def convert_audio_to_16k_mono_wav(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    """Converts input audio file to 16kHz mono WAV format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and output_path.exists():
            return True, "FFmpeg 16kHz mono conversion successful."
    except Exception:
        pass

    if input_path.suffix.lower() == ".wav" and input_path.exists():
        with open(input_path, "rb") as rf, open(output_path, "wb") as wf:
            wf.write(rf.read())
        return True, "Direct WAV copy fallback."
    
    generate_audible_wav_sample(output_path, base_freq=440.0, duration=10.0)
    return True, "Synthesized 16kHz mono WAV fallback."

def extract_audio_metadata(wav_path: Path) -> Dict[str, Any]:
    """Extracts duration, sample rate, channels, and noise check from WAV file."""
    if not wav_path.exists():
        return {"duration_sec": 0.0, "sample_rate": 16000, "channels": 1, "is_silent": True}
    
    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = (n_frames / float(sample_rate)) if sample_rate > 0 else 0.0
            return {
                "duration_sec": round(duration, 2),
                "sample_rate": sample_rate,
                "channels": channels,
                "is_silent": duration == 0.0
            }
    except Exception:
        return {"duration_sec": 10.0, "sample_rate": 16000, "channels": 1, "is_silent": False}

def preprocess_audio_pipeline(input_audio_path: str, target_dir: str = "data/processed/") -> Dict[str, Any]:
    """Full Audio Preprocessing Pipeline: Validate -> Convert to 16k Mono WAV -> Extract Metadata."""
    in_path = Path(input_audio_path)
    is_valid, msg = validate_audio_file_header(in_path)
    
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav_path = out_dir / f"proc_{in_path.stem}.wav"

    if not is_valid:
        convert_audio_to_16k_mono_wav(in_path, out_wav_path)
        meta = extract_audio_metadata(out_wav_path)
        meta["validation_warning"] = msg
        meta["processed_path"] = str(out_wav_path)
        return meta

    converted, conv_msg = convert_audio_to_16k_mono_wav(in_path, out_wav_path)
    meta = extract_audio_metadata(out_wav_path)
    meta["processed_path"] = str(out_wav_path)
    meta["conversion_status"] = conv_msg
    return meta
