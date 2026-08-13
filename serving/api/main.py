import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dialect_id.infer import infer_dialect_distribution
from serving.api.content_filter import check_content_safety
from serving.api.bhashini_adapter import convert_ulca_request_to_native, convert_native_response_to_ulca
from active_learning.annotation_queue import push_to_queue

app = FastAPI(title="Rajasthani Multi-Dialect Language API", version="1.0")

VALID_API_KEYS = {"test_key", "bhashini_key_123", "demo_key_456"}

# Tracking rate limits for ambiguous queue writes per key
AMBIENT_QUEUE_WRITES = {}

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key header.")
    return x_api_key

def get_production_model(task: str, dialect: str) -> str:
    prod_file = ROOT_DIR / "checkpoints" / task / dialect / "production.json"
    if prod_file.exists():
        with open(prod_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("run_id", f"{task}_{dialect}_prod")
    return f"{task}_{dialect}_base"

class ASRRequest(BaseModel):
    audio_path: Optional[str] = None
    text_context: Optional[str] = None
    dialect: Optional[str] = None

class MTRequest(BaseModel):
    text: str
    source_dialect: Optional[str] = None
    target_dialect: str = "hin"

class TTSRequest(BaseModel):
    text: str
    dialect: str
    backend: Optional[str] = "mms"

class DialectIDRequest(BaseModel):
    text: Optional[str] = None
    audio_path: Optional[str] = None

def check_auto_routing(text: str = None, audio_path: str = None, api_key: str = "test_key"):
    dist, top_1 = infer_dialect_distribution(text, audio_path)
    sorted_candidates = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    
    p1 = sorted_candidates[0][1]
    p2 = sorted_candidates[1][1]
    gap = p1 - p2

    conf_thresh = 0.60
    margin_thresh = 0.15

    is_ambiguous = (p1 < conf_thresh or gap < margin_thresh)

    if is_ambiguous:
        # Rate limit ambiguous queue writes (max 20 per hour per key)
        now = time.time()
        key_history = AMBIENT_QUEUE_WRITES.get(api_key, [])
        key_history = [t for t in key_history if now - t < 3600]

        suppressed = False
        if len(key_history) >= 20:
            suppressed = True
        else:
            key_history.append(now)
            AMBIENT_QUEUE_WRITES[api_key] = key_history
            # Write ambiguous event to annotation queue
            push_to_queue([{"text_dialect": text or "audio_input", "dialect": top_1, "probabilities": dist}], top_1, "ambiguous_routing")

        return True, {
            "dialect_ambiguous": True,
            "top_candidates": [c[0] for c in sorted_candidates[:3]],
            "probabilities": dist,
            "queue_writes_suppressed": suppressed
        }

    return False, top_1

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "rajasthani-lm-api", "version": "1.0"}

@app.get("/models")
def list_loaded_models():
    dialects = ["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"]
    models = {"asr": {}, "mt": {}, "tts": {}}
    for d in dialects:
        for t in ["asr", "mt", "tts"]:
            models[t][d] = get_production_model(t, d)
    return {"active_models": models}

@app.post("/dialect-id")
def dialect_identification(req: DialectIDRequest, api_key: str = Depends(verify_api_key)):
    dist, top_1 = infer_dialect_distribution(req.text, req.audio_path)
    return {"top_dialect": top_1, "probabilities": dist}

@app.post("/asr")
def asr_transcribe(req: ASRRequest, api_key: str = Depends(verify_api_key)):
    dialect = req.dialect
    if not dialect:
        is_ambiguous, routing_res = check_auto_routing(req.text_context, req.audio_path, api_key)
        if is_ambiguous:
            return JSONResponse(status_code=300, content=routing_res)
        dialect = routing_res

    prod_model = get_production_model("asr", dialect)
    sample_transcripts = {
        "mwr": "म्हारो नाम राम है।",
        "mtr": "म्हाणो घर उदयपुर में है।",
        "dhd": "जयपुर में छै।",
        "hdt": "अतरी बात सही है।",
        "mwt": "हवै सब ठीक छै।",
        "bgr": "आपणो काम हो गयो।"
    }

    transcript = sample_transcripts.get(dialect, "म्हारो नाम राम है।")
    return {
        "dialect": dialect,
        "transcript": transcript,
        "model_run_id": prod_model,
        "confidence": 0.94
    }

@app.post("/mt")
def mt_translate(req: MTRequest, api_key: str = Depends(verify_api_key)):
    source_dialect = req.source_dialect
    if not source_dialect:
        is_ambiguous, routing_res = check_auto_routing(req.text, None, api_key)
        if is_ambiguous:
            return JSONResponse(status_code=300, content=routing_res)
        source_dialect = routing_res

    prod_model = get_production_model("mt", source_dialect)
    return {
        "source_dialect": source_dialect,
        "target_language": req.target_dialect,
        "original_text": req.text,
        "translation": f"[Translated to {req.target_dialect}]: {req.text}",
        "model_run_id": prod_model
    }

@app.post("/tts")
def tts_synthesize(req: TTSRequest, api_key: str = Depends(verify_api_key)):
    # Content moderation filter check on /tts input!
    is_blocked, safety_score = check_content_safety(req.text)
    if is_blocked:
        return JSONResponse(status_code=400, content={"content_blocked": True, "reason": "Text violates safety/moderation guidelines.", "safety_score": safety_score})

    prod_model = get_production_model("tts", req.dialect)
    return {
        "dialect": req.dialect,
        "backend": req.backend,
        "text": req.text,
        "audio_b64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
        "model_run_id": prod_model
    }

@app.post("/ulca")
def bhashini_ulca_pipeline(ulca_req: Dict, api_key: str = Depends(verify_api_key)):
    native_req = convert_ulca_request_to_native(ulca_req)
    task = native_req.get("task", "asr")

    if task == "asr":
        resp = asr_transcribe(ASRRequest(dialect=native_req.get("dialect")), api_key)
    elif task == "mt":
        resp = mt_translate(MTRequest(text=native_req.get("text"), source_dialect=native_req.get("source_dialect")), api_key)
    else:
        resp = tts_synthesize(TTSRequest(text=native_req.get("text"), dialect=native_req.get("dialect")), api_key)

    if isinstance(resp, JSONResponse):
        return resp

    return convert_native_response_to_ulca(resp, task)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
