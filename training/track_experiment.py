import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def log_experiment_run(task: str, dialect: str, run_id: str, hyperparams: dict, metrics: dict, gpu_hours: float = 0.0):
    log_dir = ROOT_DIR / "logs" / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": task,
        "dialect": dialect,
        "run_id": run_id,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "gpu_hours": gpu_hours
    }

    log_file = log_dir / "experiment_history.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"[MLflow/Tracking] Logged run '{run_id}' for task '{task}' ({dialect}) to {log_file}")

if __name__ == "__main__":
    log_experiment_run("asr", "mwr", "run_test_01", {"lr": 1e-4}, {"wer": 8.5}, 4.0)
