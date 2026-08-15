import subprocess
try:
    import pytest
except ImportError:
    pytest = None

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def test_section9_backup_script_dry_run():
    script_path = ROOT_DIR / "scripts" / "backup_data.sh"
    assert script_path.exists()
    
    # Convert Windows drive path C:\path to /c/path for bash
    path_str = script_path.as_posix()
    if path_str[1:3] == ":/":
        posix_path = "/" + path_str[0].lower() + path_str[2:]
    else:
        posix_path = path_str

    res = subprocess.run(["bash", posix_path, "--dry-run"], capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback to direct script execution
        res = subprocess.run(["bash", "scripts/backup_data.sh", "--dry-run"], cwd=str(ROOT_DIR), capture_output=True, text=True)

    assert res.returncode == 0
    assert "[DRY-RUN PASS]" in res.stdout

if __name__ == "__main__":
    test_section9_backup_script_dry_run()
    print("test_section9: PASS")
