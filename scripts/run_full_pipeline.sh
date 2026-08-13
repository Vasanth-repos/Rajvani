#!/usr/bin/env bash
set -e

DIALECT="mwr"
if [ "$1" == "--dialect" ] && [ -n "$2" ]; then
    DIALECT="$2"
fi

export PYTHONIOENCODING=utf-8

if command -v python.exe &> /dev/null; then
    PY_BIN="python.exe"
elif command -v python &> /dev/null; then
    PY_BIN="python"
elif command -v python3 &> /dev/null; then
    PY_BIN="python3"
else
    PY_BIN="python"
fi

echo "============================================================"
echo " Running Full Rajasthani LM Pipeline for Dialect: $DIALECT"
echo "============================================================"

# Step 1: Validate Schema & Consent Protocol
echo "--- [Step 1/9] Validating Schemas & Consent Artifacts ---"
$PY_BIN -m data.schema.validate --dialect "$DIALECT"
$PY_BIN -m scripts.generate_consent_artifacts

# Step 2: Orthography Normalization & Split Assignment
echo "--- [Step 2/9] Normalization & Split Assignment ---"
$PY_BIN -m data.normalize_orthography --dialect "$DIALECT"
$PY_BIN -m data.splits.assign_split --dialect "$DIALECT"

# Step 3: Active Learning Priority Pool Scoring
echo "--- [Step 3/9] Active Learning Priority Pool Scoring ---"
$PY_BIN -m active_learning.score_pool --dialect "$DIALECT" --checkpoint base

# Step 4: Synthetic Augmentation Passes
echo "--- [Step 4/9] Synthetic Data Augmentation ---"
$PY_BIN -m augmentation.back_translate --dialect "$DIALECT" --input-file "data/splits/$DIALECT/train.jsonl"
$PY_BIN -m augmentation.report --dialect "$DIALECT"

# Step 5: Code-Switching & Idiom Evaluation
echo "--- [Step 5/9] Code-Switching & Idiom Bank Evaluation ---"
$PY_BIN -m codeswitch.cs_eval_set_builder --dialect "$DIALECT"
$PY_BIN -m linguistic_artifacts.idiom_mt_eval --dialect "$DIALECT"

# Step 6: LoRA Model Training & Checkpoint Promotion Gate
echo "--- [Step 6/9] Model Fine-Tuning & Promotion Gate ---"
$PY_BIN -m training.train_asr --dialect "$DIALECT"
$PY_BIN -m training.train_mt --dialect "$DIALECT"

# Step 7: Generational Drift & Limitations Generation
echo "--- [Step 7/9] Generational Drift & Limitations Report ---"
$PY_BIN -m eval.generational_drift --dialect "$DIALECT"
$PY_BIN -m eval.limitations_gen

# Step 8: Public Benchmark Leaderboard
echo "--- [Step 8/9] Benchmark Leaderboard Generation ---"
$PY_BIN -m benchmark.run_baselines --no-paid

# Step 9: Consolidated Assessment Report
echo "--- [Step 9/9] Consolidated Report Assembly ---"
$PY_BIN -m eval.report --dialect "$DIALECT"

echo "============================================================"
echo " Full Pipeline Execution Finished Successfully for '$DIALECT' ✅"
echo "============================================================"
