#!/usr/bin/env bash
set -e

echo "=== Running Automated Verification Test Suite (make check) ==="

# Check python binary
if command -v python.exe &> /dev/null; then
    PY_BIN="python.exe"
elif command -v python &> /dev/null; then
    PY_BIN="python"
elif command -v python3 &> /dev/null; then
    PY_BIN="python3"
else
    PY_BIN="python"
fi

FAILURES=0

run_section_check() {
    SECTION_NAME="$1"
    TEST_FILE="$2"
    echo -n "Checking $SECTION_NAME ... "
    if $PY_BIN -m pytest -q "$TEST_FILE" > /dev/null 2>&1; then
        echo "✅ PASS"
    else
        echo "❌ FAIL ($TEST_FILE)"
        # Print output for debugging
        $PY_BIN -m pytest "$TEST_FILE" || true
        FAILURES=$((FAILURES + 1))
    fi
}

run_section_check "Section 1 (Repo Layout & Env Setup)" "tests/test_section1.py"
run_section_check "Section 2 (Schemas, Orthography & Split Isolation)" "tests/test_section2.py"
run_section_check "Section 3 (Active Learning Priority Scorer)" "tests/test_section3.py"
run_section_check "Section 4 (Synthetic Data Augmentation)" "tests/test_section4.py"
run_section_check "Section 5 (Dialect-ID, Code-Switching & Idiom Bank)" "tests/test_section5.py"
run_section_check "Section 6 (LoRA Training & Promotion Gate)" "tests/test_section6.py"
run_section_check "Section 8 (Serving API, ULCA Adapter & Content Moderation)" "tests/test_section8.py"
run_section_check "Section 8.5 (Public Benchmark & k-Anonymity Filter)" "tests/test_section8_5.py"
run_section_check "Section 9 (Secondary Storage Backup Sync)" "tests/test_section9.py"
run_section_check "Section 10 (Telephony IVR Channel)" "tests/test_section10.py"
run_section_check "Section 11 (Evaluation Consistency Gate)" "tests/test_eval_consistency.py"
run_section_check "Section 12 (Benchmark Leakage & Verification Gate)" "tests/test_verify_benchmark.py"

echo "---------------------------------------------------------"
if [ $FAILURES -eq 0 ]; then
    echo "=== ALL SECTION CHECKS PASSED SUCCESSFULLY (12/12) ✅ ==="
    exit 0
else
    echo "=== VERIFICATION FAILED: $FAILURES section(s) regressed! ❌ ==="
    exit 1
fi
