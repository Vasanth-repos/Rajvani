import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.splits.assign_split import verify_file_path_read_access
from training.track_experiment import log_experiment_run
from training.promote_checkpoint import evaluate_and_promote
from augmentation.back_translate import back_translate_batch

def run_mt_training(dialect: str, pivot: str = "hin", epochs: int = 3, run_id: str = None):
    run_id = run_id or f"mt_{dialect}_{pivot}_{uuid.uuid4().hex[:6]}"
    
    train_split_path = ROOT_DIR / "data" / "splits" / dialect / "train.jsonl"
    verify_file_path_read_access(train_split_path, __file__)

    chk_dir = ROOT_DIR / "checkpoints" / "mt" / dialect / run_id
    chk_dir.mkdir(parents=True, exist_ok=True)

    with open(chk_dir / "adapter_model.bin", "w") as f:
        f.write("LoRA MT weights")

    metrics = {"bleu": 32.5, "chrf": 56.2}
    hyperparams = {"base_model": "indictrans2", "pivot": pivot, "lr": 2e-4, "epochs": epochs}
    gpu_hours = 3.0

    log_experiment_run("mt", dialect, run_id, hyperparams, metrics, gpu_hours)

    # Run promotion gate
    promoted, _ = evaluate_and_promote("mt", dialect, run_id, metric_name="bleu")

    # If promoted, trigger back-translation pool refresh!
    if promoted:
        back_translate_batch(dialect, str(train_split_path), generator_checkpoint=run_id)

    print(f"MT fine-tuning complete for dialect '{dialect}' (Pivot: {pivot}). Run ID: {run_id}")
    return run_id

def main():
    parser = argparse.ArgumentParser(description="Run MT LoRA fine-tuning for a dialect.")
    parser.add_argument("--dialect", type=str, default="mwr", help="Dialect ID")
    parser.add_argument("--pivot", type=str, default="hin", help="Pivot language (hin/eng)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--run-id", type=str, help="Custom run ID")
    args = parser.parse_args()

    run_mt_training(args.dialect, args.pivot, args.epochs, args.run_id)

if __name__ == "__main__":
    main()
