#!/usr/bin/env bash
set -e

echo "=== Setting up environment for Rajasthani LM pipeline ==="

# Check Python availability
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    echo "Error: Python installation not found!"
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"

# Create virtualenv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    $PYTHON_BIN -m venv venv || true
fi

# Upgrade pip & install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing required Python packages..."
    $PYTHON_BIN -m pip install --upgrade pip > /dev/null 2>&1 || true
    $PYTHON_BIN -m pip install -r requirements.txt || echo "Warning: Some packages may need manual installation depending on GPU/CPU environment."
fi

# Ensure required directory tree exists
mkdir -p configs/orthography data/raw data/validated data/synthetic data/splits \
         docs linguistic_artifacts/schema dialect_id active_learning augmentation \
         codeswitch training eval benchmark serving/api serving/ivr serving/demo_app \
         cards/model_cards cards/dataset_cards tests scripts checkpoints logs

echo "=== Environment setup complete. ==="
