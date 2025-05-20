#!/usr/bin/env python
"""
Strategy Runner with RegimeAdaptiveStrategy Patch

This is a wrapper script that applies the regime_adaptive_patch
and then runs the original strategy_runner.py script.

Use this script exactly as you would the original strategy_runner.py.
"""

import os
import sys
import subprocess

# Import the patch module and apply it
try:
    from regime_adaptive_patch import patch_regime_adaptive_strategy
    # Apply the patch
    patch_regime_adaptive_strategy()
    print("RegimeAdaptiveStrategy patch applied successfully")
except ImportError:
    print("Warning: regime_adaptive_patch.py not found. Continuing without the patch.")
except Exception as e:
    print(f"Warning: Error applying RegimeAdaptiveStrategy patch: {e}")

# Import and run the original strategy_runner
try:
    import strategy_runner
    
    if __name__ == "__main__":
        # Pass all command line arguments to the original script
        strategy_runner.main()
except Exception as e:
    print(f"Error running strategy_runner: {e}")
    sys.exit(1)
