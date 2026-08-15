try:
    import pytest
except ImportError:
    pytest = None

from pathlib import Path
import sys
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from serving.api.main import app
from serving.api.content_filter import check_content_safety

client = TestClient(app)

def test_section8_api_key_auth_and_health():
    # Health check is open
    response_health = client.get("/health")
    assert response_health.status_code == 200

    # Protected endpoint requires key
    response_no_key = client.post("/api/pipeline/run", json={"dialect": "MWR"})
    assert response_no_key.status_code == 401

    # Valid key passes
    response_valid = client.post("/api/pipeline/run", json={"dialect": "MWR"}, headers={"X-API-Key": "test_key"})
    assert response_valid.status_code == 200

def test_section8_provider_status_and_dialects():
    resp_dialects = client.get("/api/dialects")
    assert resp_dialects.status_code == 200
    assert "dialects" in resp_dialects.json()

    resp_providers = client.get("/api/providers/status")
    assert resp_providers.status_code == 200
    assert "providers" in resp_providers.json()

def test_section8_content_filter_on_tts():
    # Safe text passes
    safe_blocked, safe_score = check_content_safety("म्हारो नाम राम है।")
    assert safe_blocked is False

    # Unsafe text blocked
    unsafe_text = "This contains abusive_term text"
    unsafe_blocked, unsafe_score = check_content_safety(unsafe_text)
    assert unsafe_blocked is True

    # /api/tts returns content_blocked for unsafe text
    resp_tts = client.post("/api/tts", json={"text": unsafe_text, "dialect": "MWR"}, headers={"X-API-Key": "test_key"})
    assert resp_tts.status_code == 200
    assert resp_tts.json().get("content_blocked") is True

if __name__ == "__main__":
    test_section8_api_key_auth_and_health()
    test_section8_provider_status_and_dialects()
    test_section8_content_filter_on_tts()
    print("test_section8: PASS")
