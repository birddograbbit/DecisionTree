# macOS Compatibility Fix Documentation

## Issue Summary
The DecisionTree project encountered PyTorch compatibility issues on macOS ARM (M1/M2/M3) architecture, resulting in segmentation faults during transformer model tests.

## Root Cause
PyTorch 2.7.1 on Apple Silicon has known issues with:
1. Multi-head attention mechanisms
2. Linear algebra operations
3. Threading conflicts between PyTorch and system libraries

## Fixes Implemented (2025-06-28)

### 1. Module Discovery Fix
- **File**: `setup.py`
- **Purpose**: Make the project installable as a Python package
- **Usage**: `pip install -e .`

### 2. macOS Compatibility Patches
- **File**: `src/models/transformer/macos_patches.py`
- **Features**:
  - Automatic detection of macOS ARM architecture
  - Single-threaded execution enforcement
  - Safe device selection (CPU-only)
  - Environment variable configuration

### 3. Model Updates
- **Files**: `transformer_model.py`, `transformer_wrapper.py`
- **Changes**:
  - Import and apply macOS patches before PyTorch
  - Reduced batch sizes for stability
  - Disabled multiprocessing in DataLoader
  - Added error handling for runtime issues

### 4. Test Runner Script
- **File**: `run_tests_macos.sh`
- **Purpose**: Set all required environment variables
- **Usage**: `chmod +x run_tests_macos.sh && ./run_tests_macos.sh`

## Usage Instructions

### Quick Fix (Recommended)
```bash
# Install package in development mode
pip install -e .

# Run tests with macOS compatibility
./run_tests_macos.sh
```

### Manual Setup
```bash
# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install and run tests
pip install -e .
pytest tests/test_transformer_integration.py -v
```

## Performance Considerations
- Single-threaded execution reduces performance but ensures stability
- Batch sizes are automatically reduced on macOS
- For production use, consider Linux or Docker deployment

## Alternative Solutions
1. **PyTorch Downgrade**: `pip install torch==2.2.2`
2. **Docker**: Use containerized Linux environment
3. **Cloud Development**: Use Linux-based cloud instances

## Verification
After applying fixes, tests should pass without segmentation faults:
```bash
pytest tests/test_transformer_integration.py -v  # Should show 3/3 PASSED
pytest tests/test_hybrid_strategy.py -v         # Should show all tests PASSED
```

## Future Improvements
- Monitor PyTorch releases for native Apple Silicon support improvements
- Consider implementing MPS (Metal Performance Shaders) support when stable
- Optimize batch processing for macOS-specific constraints
