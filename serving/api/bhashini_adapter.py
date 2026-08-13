"""
BHASHINI ULCA Adapter (serving/api/bhashini_adapter.py)
------------------------------------------------------
Pinned ULCA Schema Version: v2.0
Schema Resolved & Verified Date: 2026-08-13
Reference Docs: https://bhashini.gov.in/ulca/docs/api

Wraps native /asr, /mt, /tts endpoints in ULCA pipeline format.
"""

import json
from typing import Dict, Any

def convert_ulca_request_to_native(ulca_req: Dict[str, Any]) -> Dict[str, Any]:
    """Converts ULCA pipelineTasks payload to native parameters."""
    tasks = ulca_req.get("pipelineTasks", [])
    if not tasks:
        return {}

    first_task = tasks[0]
    task_type = first_task.get("taskType")
    config = first_task.get("config", {})

    source_lang = config.get("language", {}).get("sourceLanguage")
    target_lang = config.get("language", {}).get("targetLanguage")

    if task_type == "asr":
        return {
            "task": "asr",
            "dialect": source_lang,
            "audio_content": first_task.get("inputData", {}).get("audio", [{}])[0].get("audioContent")
        }
    elif task_type == "translation":
        return {
            "task": "mt",
            "source_dialect": source_lang,
            "target_dialect": target_lang,
            "text": first_task.get("inputData", {}).get("input", [{}])[0].get("source")
        }
    elif task_type == "tts":
        return {
            "task": "tts",
            "dialect": source_lang,
            "text": first_task.get("inputData", {}).get("input", [{}])[0].get("source")
        }
    return {}

def convert_native_response_to_ulca(native_resp: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """Converts native response into ULCA pipelineResponse format."""
    if task_type == "asr":
        return {
            "pipelineResponse": [
                {
                    "taskType": "asr",
                    "output": [
                        {
                            "source": native_resp.get("transcript", "")
                        }
                    ]
                }
            ]
        }
    elif task_type == "translation":
        return {
            "pipelineResponse": [
                {
                    "taskType": "translation",
                    "output": [
                        {
                            "target": native_resp.get("translation", "")
                        }
                    ]
                }
            ]
        }
    elif task_type == "tts":
        return {
            "pipelineResponse": [
                {
                    "taskType": "tts",
                    "audio": [
                        {
                            "audioContent": native_resp.get("audio_b64", "")
                        }
                    ]
                }
            ]
        }
    return {"pipelineResponse": []}
