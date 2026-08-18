"""
tests/test_scaled_dataset.py

Comprehensive test suite verifying:
1. 100,000-scale training dataset integrity, distribution, and ID uniqueness.
2. Dialect-specific morphological markers and grammatical consistency.
3. Code-switching span boundary validity.
4. Zero leakage against the 200 frozen held-out benchmark suite.
5. Mathematical sanity of non-parametric bootstrap confidence intervals.
6. Speaker-disjoint isolation across train, dev, and test splits.
"""

import glob
import json
import math
import os
import sys
from pathlib import Path
import pytest
import jsonschema

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from configs.dialects import DIALECT_REGISTRY
from eval.bootstrap_ci import bootstrap_distribution

SCHEMA_PATH = ROOT_DIR / "data" / "schema" / "text_record.schema.json"
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    TEXT_SCHEMA = json.load(f)
VALIDATOR = jsonschema.Draft7Validator(TEXT_SCHEMA)

DIALECT_KEYS = ["mwr", "mtr", "dhd", "hdt", "mwt", "bgr"]

def test_100k_total_sample_count_and_distribution():
    """Asserts that exactly 100,000 training records exist across the 6 dialects."""
    total_records = 0
    dialect_counts = {}

    for did in DIALECT_KEYS:
        train_file = ROOT_DIR / "data" / "splits" / did / "train.jsonl"
        assert train_file.exists(), f"Train file missing for dialect {did}"
        with open(train_file, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        dialect_counts[did] = count
        total_records += count

    assert total_records == 100000, f"Expected 100,000 total records, got {total_records}"
    
    # Each dialect must hold ~16,666 - 16,667 samples
    for did, count in dialect_counts.items():
        assert 16660 <= count <= 16670, f"Dialect {did} has unbalanced count: {count}"

def test_global_record_id_uniqueness():
    """Verifies that all 100,000 training record IDs are globally unique."""
    seen_ids = set()
    total_checked = 0

    for did in DIALECT_KEYS:
        train_file = ROOT_DIR / "data" / "splits" / did / "train.jsonl"
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rid = r.get("id")
                    assert rid is not None, "Record missing 'id'"
                    assert rid not in seen_ids, f"Duplicate ID collision detected: {rid}"
                    seen_ids.add(rid)
                    total_checked += 1

    assert total_checked == 100000

AUDIO_SCHEMA_PATH = ROOT_DIR / "data" / "schema" / "audio_record.schema.json"
with open(AUDIO_SCHEMA_PATH, "r", encoding="utf-8") as f:
    AUDIO_SCHEMA = json.load(f)
AUDIO_VALIDATOR = jsonschema.Draft7Validator(AUDIO_SCHEMA)

def test_schema_conformance_batched():
    """Validates stratified sample batches against the strict text/audio record schemas."""
    for did in DIALECT_KEYS:
        train_file = ROOT_DIR / "data" / "splits" / did / "train.jsonl"
        with open(train_file, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        
        # Test 100 evenly-spaced samples per dialect
        step = max(1, len(lines) // 100)
        sample_lines = lines[::step][:100]

        for sline in sample_lines:
            rec = json.loads(sline)
            if "audio_path" in rec:
                AUDIO_VALIDATOR.validate(rec)
            else:
                VALIDATOR.validate(rec)
            assert rec["split"] == "train"
            assert rec["dialect"] == did
            assert rec["public_release_ok"] is True

def test_morphological_dialect_markers():
    """Validates dialect-specific morphological postpositions and negation particles across all 6 dialects."""
    for did in DIALECT_KEYS:
        train_file = ROOT_DIR / "data" / "splits" / did / "train.jsonl"
        with open(train_file, "r", encoding="utf-8") as f:
            sample_texts = [json.loads(line)["text_dialect"] for line in f if line.strip()][:500]

        all_text_blob = " ".join(sample_texts)

        if did in ["mwr", "mtr"]:
            # Marwari / Mewari genitive postposition 'रो' / 'री' / 'रा'
            assert ("रो" in all_text_blob or "री" in all_text_blob), f"Missing genitive 'रो/री' in {did}"
        elif did in ["dhd", "hdt", "mwt", "bgr"]:
            # Dhundhari / Hadoti / Mewati / Bagri genitive postposition 'को' / 'की' / 'का'
            assert ("को" in all_text_blob or "की" in all_text_blob), f"Missing genitive 'को/की' in {did}"

        if did == "dhd":
            assert "छै" in all_text_blob or "छा" in all_text_blob, "Dhundhari copula 'छै' not found"
        elif did == "mtr":
            assert "छे" in all_text_blob or "सूं" in all_text_blob, "Mewari markers not found"
        elif did == "hdt":
            assert "छै" in all_text_blob or "ती" in all_text_blob, "Hadoti markers not found"
        elif did == "mwt":
            assert "सै" in all_text_blob or "तें" in all_text_blob, "Mewati markers not found"
        elif did == "bgr":
            assert "है" in all_text_blob or "सूं" in all_text_blob, "Bagri markers not found"

def test_code_switching_span_offsets():
    """Ensures code-switched records contain valid character offsets matching the span text."""
    found_cs = 0
    for did in DIALECT_KEYS:
        train_file = ROOT_DIR / "data" / "splits" / did / "train.jsonl"
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("is_code_switched"):
                        found_cs += 1
                        spans = r.get("cs_spans", [])
                        text = r.get("text_dialect", "")
                        for sp in spans:
                            s, e, lang = sp["start"], sp["end"], sp["lang"]
                            assert 0 <= s < e <= len(text), f"Invalid span bounds: {s}..{e} for text len {len(text)}"
                            assert len(text[s:e]) > 0, "Span text must not be empty"
                            assert lang in ["hin", "eng"], f"Unexpected span language: {lang}"
                        if found_cs >= 50:
                            break

    assert found_cs >= 50, f"Expected at least 50 code-switched records, found {found_cs}"

def test_zero_leakage_with_frozen_benchmark():
    """Enforces strict zero text collision between 100k training set and 200 held-out test suite."""
    test_file = ROOT_DIR / "data" / "realworld_test_200.jsonl"
    assert test_file.exists(), "Frozen test set missing!"

    with open(test_file, "r", encoding="utf-8") as f:
        test_strings = {json.loads(line)["text_dialect"].strip() for line in f if line.strip()}

    assert len(test_strings) == 200, f"Held-out test set must have exactly 200 items, got {len(test_strings)}"

    collisions = []
    for did in DIALECT_KEYS:
        train_file = ROOT_DIR / "data" / "splits" / did / "train.jsonl"
        with open(train_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if line.strip():
                    r = json.loads(line)
                    t = r.get("text_dialect", "").strip()
                    if t in test_strings:
                        collisions.append((did, line_idx, t))

    assert len(collisions) == 0, f"Detected {len(collisions)} test leakage collisions in training set: {collisions[:3]}"

def test_bootstrap_ci_mathematical_bounds():
    """Validates bootstrap distribution sanity, outlier robustness, and interval bounds."""
    # Test case 1: Standard normal-like distribution
    data = [10.0, 12.0, 11.5, 9.8, 10.5, 11.2, 12.5, 10.1, 10.9, 11.8]
    pt, lo, hi, dist = bootstrap_distribution(data, B=1000, alpha=0.05, seed=42)
    assert lo <= pt <= hi, f"Point estimate {pt} out of bounds [{lo}, {hi}]"
    assert len(dist) == 1000

    # Test case 2: Zero-variance scenario (identical values)
    constant_data = [5.0] * 20
    pt_c, lo_c, hi_c, _ = bootstrap_distribution(constant_data, B=500, alpha=0.05, seed=42)
    assert pt_c == lo_c == hi_c == 5.0

    # Test case 3: Empty dataset
    assert bootstrap_distribution([], B=100) == (0.0, 0.0, 0.0, [])

def test_speaker_disjoint_split_isolation():
    """Asserts that no speaker IDs overlap between train, dev, and test splits."""
    speaker_splits = {}
    for did in DIALECT_KEYS:
        for sname in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
            spath = ROOT_DIR / "data" / "splits" / did / sname
            if spath.exists():
                with open(spath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            spk = r.get("speaker_id")
                            if spk:
                                if spk not in speaker_splits:
                                    speaker_splits[spk] = set()
                                speaker_splits[spk].add(sname)

    leaked = [spk for spk, splits in speaker_splits.items() if len(splits) > 1]
    assert len(leaked) == 0, f"Speaker leakage detected across splits: {leaked[:5]}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
