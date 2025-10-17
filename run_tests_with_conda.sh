#!/bin/bash
set -e

# Script to run tests in a temporary conda environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="autoCLI_config_test_$(date +%s)"

echo "=========================================="
echo "Creating temporary conda environment: $ENV_NAME"
echo "=========================================="

# Create conda environment
conda create -n "$ENV_NAME" python=3.9 -y

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ""
echo "=========================================="
echo "Installing package and dependencies"
echo "=========================================="

# Install the package in editable mode with dev dependencies
cd "$SCRIPT_DIR"
pip install -e ".[dev]"

echo ""
echo "=========================================="
echo "Running tests"
echo "=========================================="

# Run pytest
pytest tests/ -v --cov=autoCLI_config --cov-report=term-missing

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Cleaning up"
echo "=========================================="

# Deactivate conda environment
conda deactivate

# Remove the temporary environment
conda env remove -n "$ENV_NAME" -y

echo ""
echo "=========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ Tests passed successfully"
else
    echo "✗ Tests failed with exit code: $TEST_EXIT_CODE"
fi
echo "=========================================="

exit $TEST_EXIT_CODE
