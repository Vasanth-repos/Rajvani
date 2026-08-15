import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def test_section9_backup_script_dry_run():
    # 1. Test cross-platform Python backup runner
    py_script = ROOT_DIR / "scripts" / "backup_data.py"
    assert py_script.exists()
    res_py = subprocess.run([sys.executable, str(py_script), "--dry-run"], capture_output=True, text=True, timeout=10)
    assert res_py.returncode == 0
    assert "[DRY-RUN PASS]" in res_py.stdout

    # 2. Test shell backup script exists
    sh_script = ROOT_DIR / "scripts" / "backup_data.sh"
    assert sh_script.exists()

if __name__ == "__main__":
    test_section9_backup_script_dry_run()
    print("test_section9: PASS")
