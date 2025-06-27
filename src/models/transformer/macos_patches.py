"""
macOS ARM (M1/M2/M3) compatibility patches for PyTorch.

This module provides patches to fix segmentation faults that occur
when running PyTorch on Apple Silicon due to threading and memory
management issues.
"""

import platform
import torch
import os
import warnings
import logging

logger = logging.getLogger(__name__)


def is_macos_arm():
    """Check if running on macOS with ARM processor."""
    return (
        platform.system() == 'Darwin' and 
        platform.processor() in ['arm', 'arm64', 'aarch64']
    )


def apply_macos_patches():
    """
    Apply patches for macOS ARM compatibility.
    
    This function sets various environment variables and PyTorch settings
    to prevent segmentation faults on Apple Silicon Macs.
    """
    if not is_macos_arm():
        return
    
    logger.info("Detected macOS ARM architecture, applying compatibility patches...")
    
    # Force single-threaded execution for various libraries
    thread_settings = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    }
    
    for key, value in thread_settings.items():
        os.environ[key] = value
    
    # PyTorch specific settings
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    # Disable MPS backend if it causes issues
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Disable memory profiler if it's causing issues
    os.environ['MEMORY_PROFILER_DISABLE'] = '1'
    
    # Log the applied settings
    logger.info("Applied macOS ARM patches:")
    for key, value in thread_settings.items():
        logger.info(f"  {key}={value}")
    logger.info(f"  PyTorch threads: {torch.get_num_threads()}")
    logger.info(f"  PyTorch interop threads: {torch.get_num_interop_threads()}")
    
    warnings.warn(
        "Running on macOS ARM with compatibility patches. "
        "Performance may be reduced due to single-threaded execution. "
        "For production use, consider running on Linux or using Docker.",
        UserWarning
    )


def get_safe_device():
    """
    Get a safe device for PyTorch operations on macOS ARM.
    
    Returns:
        torch.device: CPU device on macOS ARM, otherwise best available device.
    """
    if is_macos_arm():
        # Force CPU usage on macOS ARM to avoid MPS issues
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def safe_multiprocessing_start():
    """
    Set safe multiprocessing start method for macOS.
    
    This prevents issues with forking on macOS.
    """
    if platform.system() == 'Darwin':
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass


# Auto-apply patches when imported
apply_macos_patches()
safe_multiprocessing_start()
