try:
    import pytest
except ImportError:
    pytest = None

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from serving.ivr.twilio_app import handle_incoming_call

def test_section10_ivr_telephony_channel():
    # IVR disabled by default
    resp_disabled = handle_incoming_call("+919829000000", "sample.wav")
    assert resp_disabled["status"] == "disabled"

if __name__ == "__main__":
    test_section10_ivr_telephony_channel()
    print("test_section10: PASS")
