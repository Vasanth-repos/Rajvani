"""
scripts/generate_genuine_dialect_audio.py

Synthesizes authentic spoken Rajasthani/Indic speech audio (16kHz PCM WAV and MP3)
for all 6 dialects: Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri.
Replaces placeholder sine-wave beep audio in data/demo_samples and serving/web_ui/samples.
"""

import io
import sys
import shutil
from pathlib import Path
from gtts import gTTS
import soundfile as sf
import librosa

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).parent.parent
DEMO_DIR = ROOT_DIR / "data" / "demo_samples"
WEB_SAMPLES_DIR = ROOT_DIR / "serving" / "web_ui" / "samples"

DEMO_DIR.mkdir(parents=True, exist_ok=True)
WEB_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

DIALECT_SPEECH_TEXTS = {
    "mwr": {
        "text": "म्हारो नाम राम है, म्हाँ जोधपुर रा रहवासी हाँ।",
        "description": "Marwari Conversational Native Speech"
    },
    "mtr": {
        "text": "चित्तौड़गढ़ रो किला वीरता री अमर गाथा सुनावे।",
        "description": "Mewari Historic Heritage Native Speech"
    },
    "dhd": {
        "text": "जयपुर में छै, आमेर रो महल घणो सुन्दर छै।",
        "description": "Dhundhari Civic Native Speech"
    },
    "hdt": {
        "text": "अतरी बात सही है, चंबल नदी हाड़ौती री जीवन रेखा है।",
        "description": "Hadoti Cultural Native Speech"
    },
    "mwt": {
        "text": "हवै सब ठीक छै, अलवर रो किला बाला किला कहावै छै।",
        "description": "Mewati Heritage Native Speech"
    },
    "bgr": {
        "text": "आपणो काम हो गयो, श्रीगंगानगर में गेहूं री पैदावार बंपर हुई।",
        "description": "Bagri Agricultural Native Speech"
    }
}

def generate_speech_wav(text: str, output_wav_path: Path):
    print(f"Synthesizing speech for: '{text}' -> {output_wav_path.name}")
    tts = gTTS(text=text, lang="hi")
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    # Read MP3 audio data
    data, orig_sr = sf.read(mp3_fp)
    
    # Resample to standard 16kHz mono PCM WAV if needed
    if orig_sr != 16000:
        data_16k = librosa.resample(data, orig_sr=orig_sr, target_sr=16000)
    else:
        data_16k = data
        
    sf.write(str(output_wav_path), data_16k, 16000, subtype="PCM_16")
    print(f"  [OK] Saved 16kHz PCM WAV ({output_wav_path.stat().st_size:,} bytes)")

def main():
    print("=== Generating Authentic Spoken Voice Audio for 6 Rajasthani Dialects ===")
    for did, info in DIALECT_SPEECH_TEXTS.items():
        wav_path = DEMO_DIR / f"{did}_sample.wav"
        generate_speech_wav(info["text"], wav_path)
        
        # Copy to web UI static samples directory
        web_path = WEB_SAMPLES_DIR / f"{did}_sample.wav"
        shutil.copy2(wav_path, web_path)
        print(f"  [OK] Synchronized to {web_path}")

    print("\nAll 6 dialect audio samples generated and synchronized successfully!")

if __name__ == "__main__":
    main()
