import pytest
from pathlib import Path
import sys
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from serving.api.main import app, AMBIENT_QUEUE_WRITES
from serving.api.content_filter import check_content_safety

client = TestClient(app)

def test_section8_api_key_auth_and_health():
    # Health check is open
    response_health = client.get("/health")
    assert response_health.status_code == 200

    # Endpoints requiring key return 401 without header
    response_no_key = client.post("/asr", json={"dialect": "mwr"})
    assert response_no_key.status_code == 401

    # Valid key passes
    response_valid = client.post("/asr", json={"dialect": "mwr"}, headers={"X-API-Key": "test_key"})
    assert response_valid.status_code == 200

def test_section8_ambiguous_routing_and_rate_limiting():
    # Omitted dialect parameter triggers auto-routing. On ambiguous text (low top-1 / small margin), returns 300 multi-choice
    resp_ambiguous = client.post("/asr", json={"text_context": "ambiguous code switched text"}, headers={"X-API-Key": "test_key"})
    assert resp_ambiguous.status_code in [200, 300]
    if resp_ambiguous.status_code == 300:
        data = resp_ambiguous.json()
        assert data.get("dialect_ambiguous") is True
        assert "probabilities" in data

    # Test rate limit on ambiguous queue writes per API key (20/hr max)
    AMBIENT_QUEUE_WRITES["rate_test_key"] = [1000000000.0] * 20
    resp_suppressed = client.post("/asr", json={"text_context": "ambiguous text"}, headers={"X-API-Key": "rate_test_key"})
    if resp_suppressed.status_code == 300:
        assert resp_suppressed.json().get("queue_writes_suppressed") is True

def test_section8_content_filter_on_tts():
    # Safe text passes
    safe_blocked, safe_score = check_content_safety("म्हारो नाम राम है।")
    assert safe_blocked is False

    # Unsafe text blocked
    unsafe_text = "This contains abusive_term text"
    unsafe_blocked, unsafe_score = check_content_safety(unsafe_text)
    assert unsafe_blocked is True

    # /tts endpoint returns 400 content_blocked for unsafe text
    resp_tts = client.post("/tts", json={"text": unsafe_text, "dialect": "mwr"}, headers={"X-API-Key": "test_key"})
    assert resp_tts.status_code == 400
    assert resp_tts.json().get("content_blocked") is True

    # /asr is unaffected by content filter
    resp_asr = client.post("/asr", json={"text_context": unsafe_text, "dialect": "mwr"}, headers={"X-API-Key": "test_key"})
    assert resp_asr.status_code == 200
