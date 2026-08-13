import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from active_learning.annotation_queue import push_to_queue

# Feature flag check helper
def is_ivr_enabled() -> bool:
    config_file = ROOT_DIR / "configs" / "pipeline.yaml"
    if config_file.exists():
        import yaml
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("ivr", {}).get("enabled", False)
    return False

def handle_incoming_call(caller_id: str, audio_payload_path: str, dtmf_input: str = None, retry_count: int = 0):
    if not is_ivr_enabled():
        return {
            "status": "disabled",
            "message": "IVR channel is feature-flagged off (configs/pipeline.yaml: ivr.enabled: false)."
        }

    # Simulate /asr call response
    if dtmf_input == "1":
        explicit_dialect = "mwr"
    elif dtmf_input == "2":
        explicit_dialect = "mtr"
    else:
        explicit_dialect = None

    if not explicit_dialect:
        # First turn: simulate dialect_ambiguous response from /asr
        if retry_count >= 2:
            # Graceful failure turn after 2 failed retries
            return {
                "status": "failed_gracefully",
                "action": "play_pre_recorded_audio",
                "audio_clip": "prompts/graceful_failure_hindi.wav",
                "message": "We couldn't understand the dialect. Please try again later or contact support."
            }

        # Log ambiguous call event to annotation queue
        push_to_queue([{"audio_path": audio_payload_path, "caller_id": caller_id}], "mwr", "ivr")

        return {
            "status": "dialect_ambiguous",
            "action": "play_voice_disambiguation_prompt",
            "prompt_audio": "prompts/disambiguate_hindi_mwr_mtr.wav",
            "dtmf_menu": {"1": "Marwari", "2": "Mewari"},
            "retry_count": retry_count + 1
        }

    # Explicit dialect resolved
    return {
        "status": "success",
        "resolved_dialect": explicit_dialect,
        "transcript": "म्हारो काम हो गयो।",
        "translation": "मेरा काम हो गया।",
        "audio_out": "audio_response.wav"
    }

def main():
    parser = argparse.ArgumentParser(description="Simulate Twilio/Exotel IVR call handling.")
    parser.add_argument("--caller-id", type=str, default="+919829000000", help="Caller phone number")
    parser.add_argument("--audio", type=str, default="sample_call.wav", help="Audio payload path")
    parser.add_argument("--dtmf", type=str, help="DTMF digit pressed (1=Marwari, 2=Mewari)")
    parser.add_argument("--retry", type=int, default=0, help="Retry counter")
    args = parser.parse_args()

    resp = handle_incoming_call(args.caller_id, args.audio, args.dtmf, args.retry)
    print("IVR Call Response:")
    print(json.dumps(resp, indent=2))

if __name__ == "__main__":
    main()
