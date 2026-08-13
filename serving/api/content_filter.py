import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

UNSAFE_KEYWORDS = ["hatespeech", "abusive_term", "violence", "defamatory_word"]

def check_content_safety(text: str, threshold: float = 0.70):
    """
    Evaluates text for unsafe/abusive content for /tts input.
    Returns: (is_blocked, score)
    """
    if not text:
        return False, 0.0

    text_lower = text.lower()
    unsafe_match_count = sum(1 for kw in UNSAFE_KEYWORDS if kw in text_lower)
    
    score = 0.0
    if unsafe_match_count > 0:
        score = 0.95
    
    is_blocked = (score >= threshold)
    if is_blocked:
        log_moderation_event(text)

    return is_blocked, score

def log_moderation_event(raw_text: str):
    log_dir = ROOT_DIR / "serving" / "api"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "moderation_log.jsonl"

    text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_hash": text_hash,
        "action": "content_blocked"
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Content Filter] Blocked unsafe /tts input (hash: {text_hash[:8]}). Logged to {log_file}")

if __name__ == "__main__":
    blocked, score = check_content_safety("This contains abusive_term text", 0.70)
    print(f"Safety check result: Blocked={blocked}, Score={score}")
