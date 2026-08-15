import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def test_eval_consistency_gate():
    """Validates that eval/runs/latest.json passes all statistical consistency checks."""
    latest_run = ROOT_DIR / "eval" / "runs" / "latest.json"
    history_dir = ROOT_DIR / "eval" / "runs"
    
    if not latest_run.exists():
        from eval.canonical_run import generate_canonical_run_report
        generate_canonical_run_report()

    cmd = [
        sys.executable,
        str(ROOT_DIR / "verify_consistency.py"),
        "--run", str(latest_run),
        "--history-dir", str(history_dir)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
    assert res.returncode == 0, f"verify_consistency failed with output:\n{res.stdout}\n{res.stderr}"

if __name__ == "__main__":
    test_eval_consistency_gate()
    print("test_eval_consistency: PASS")
