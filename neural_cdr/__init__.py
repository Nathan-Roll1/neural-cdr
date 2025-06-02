"""
Neural Continuous-Time Deconvolutional Regression (Neural CDR)
=============================================================

A GPU-accelerated PyTorch implementation of continuous-time deconvolutional 
regression with learned neural impulse response functions.
"""

try:
    # Try relative imports (package structure)
    from .neural_cdr import (
        run_neural_cdr,
        CDRDataset,
        ImprovedNeuralIRF,
        ImprovedBlockLinearCDR,
        prepare_reading_data_for_cdr,
        train_model,
        compute_r2
    )

    from .utils_windows import (
        create_windows_numba,
        create_windows_vectorized,
        NUMBA_AVAILABLE
    )

    from .visualize import visualize_results
except ImportError:
    # Fall back to absolute imports (flat structure)
    from neural_cdr import (
        run_neural_cdr,
        CDRDataset,
        ImprovedNeuralIRF,
        ImprovedBlockLinearCDR,
        prepare_reading_data_for_cdr,
        train_model,
        compute_r2
    )

    from utils_windows import (
        create_windows_numba,
        create_windows_vectorized,
        NUMBA_AVAILABLE
    )

    from visualize import visualize_results

__version__ = "1.0.0"
__author__ = "Nathan Roll"
__email__ = "nroll@stanford.edu"

__all__ = [
    # Main function
    'run_neural_cdr',
    
    # Core classes
    'CDRDataset',
    'ImprovedNeuralIRF', 
    'ImprovedBlockLinearCDR',
    
    # Functions
    'prepare_reading_data_for_cdr',
    'train_model',
    'compute_r2',
    'visualize_results',
    
    # Window processing
    'create_windows_numba',
    'create_windows_vectorized',
    'NUMBA_AVAILABLE'
]