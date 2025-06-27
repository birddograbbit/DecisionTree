#!/bin/bash
# Test runner script with macOS compatibility settings

echo "Setting up environment for macOS compatibility..."

# Set threading environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Disable memory profiler that's causing secondary crashes
export MEMORY_PROFILER_DISABLE=1

# Enable PyTorch MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Environment variables set:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  VECLIB_MAXIMUM_THREADS=$VECLIB_MAXIMUM_THREADS"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=$PYTORCH_ENABLE_MPS_FALLBACK"
echo ""

# Install the package in development mode if not already installed
if ! pip show decisiontree-strategy >/dev/null 2>&1; then
    echo "Installing package in development mode..."
    pip install -e .
fi

echo "Running tests..."

# Run the specific failing tests
echo "Testing transformer integration..."
pytest tests/test_transformer_integration.py -v -s --tb=short

echo ""
echo "Testing hybrid strategy..."
pytest tests/test_hybrid_strategy.py -v -s --tb=short

# Run all tests if the specific ones pass
if [ $? -eq 0 ]; then
    echo ""
    echo "Running all tests..."
    pytest tests/ -v --tb=short
fi

echo ""
echo "Tests completed. Check output above for results."
