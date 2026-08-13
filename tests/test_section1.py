import os
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def test_section1_layout_and_env():
    expected_files = [
        "README.md",
        "BUDGET.md",
        "LICENSES.md",
        "LIMITATIONS.md",
        "Makefile",
        "requirements.txt",
        "configs/dialects.yaml",
        "configs/pipeline.yaml",
        "docs/CONSENT_PROTOCOL.md"
    ]
    for rel_path in expected_files:
        p = ROOT_DIR / rel_path
        assert p.exists(), f"Section 1 Layout Error: Missing file '{rel_path}'"
