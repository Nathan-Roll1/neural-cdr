"""
Example Neural CDR Analysis
===========================

This script demonstrates how to use Neural CDR on reading time data.
"""

# If running from examples/ directory, add parent to path
# This is only needed if you haven't installed the package
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from neural_cdr import run_neural_cdr


def main():
    """Run a complete Neural CDR analysis"""
    
    # =========================================================================
    # LOAD YOUR DATA
    # =========================================================================
    
    # Example with CSV files:
    # X = pd.read_csv('path/to/predictors.csv')
    # y = pd.read_csv('path/to/readingtimes.csv')
    
    # Example with pickle files:
    # X = pd.read_pickle('dundee_predictors.pkl')
    # y = pd.read_pickle('dundee_readingtimes.pkl')
    
    # For this example, let's create synthetic data
    print("Creating synthetic data for demonstration...")
    n_obs = 10000
    n_sentences = 100
    
    # Create synthetic predictor data
    X = pd.DataFrame({
        'sentid': np.repeat(np.arange(n_sentences), n_obs // n_sentences),
        'sentpos': np.tile(np.arange(n_obs // n_sentences), n_sentences),
        'wordlen': np.random.poisson(5, n_obs),
        'freq': np.random.normal(0, 1, n_obs),
        'surprisal': np.random.gamma(2, 2, n_obs),
        'gpt2gram': np.random.normal(5, 2, n_obs)
    })
    
    # Create synthetic reading time data
    # Simple generative model: RT = baseline + effects + noise
    baseline = 200
    wordlen_effect = X['wordlen'] * 10
    freq_effect = -X['freq'] * 15
    surprisal_effect = X['surprisal'] * 8
    
    # Add some spillover effects
    surprisal_spillover = np.concatenate([[0], surprisal_effect[:-1].values]) * 0.3
    
    y = pd.DataFrame({
        'sentid': X['sentid'],
        'sentpos': X['sentpos'],
        'fdur': baseline + wordlen_effect + freq_effect + surprisal_effect + 
                surprisal_spillover + np.random.normal(0, 30, n_obs)
    })
    
    # Ensure positive reading times
    y['fdur'] = np.maximum(y['fdur'], 50)
    
    print(f"Created {n_obs} observations across {n_sentences} sentences")
    
    # =========================================================================
    # RUN NEURAL CDR ANALYSIS
    # =========================================================================
    
    # Basic analysis with default parameters
    print("\n" + "="*80)
    print("RUNNING BASIC ANALYSIS WITH DEFAULT PARAMETERS")
    print("="*80)
    
    results_basic = run_neural_cdr(
        X, y,
        predictor_cols=['surprisal'],
        response_col='fdur',
        visualize=True
    )
    
    print(f"\nBasic analysis complete!")
    print(f"Final validation R²: {results_basic['results']['r2']:.4f}")
    
    # =========================================================================
    # ADVANCED ANALYSIS WITH CUSTOM PARAMETERS
    # =========================================================================
    
    print("\n" + "="*80)
    print("RUNNING ADVANCED ANALYSIS WITH CUSTOM PARAMETERS")
    print("="*80)
    
    results_advanced = run_neural_cdr(
        X, y,
        predictor_cols=['surprisal', 'wordlen', 'freq'],
        response_col='fdur',
        # Model architecture
        history_length=7,          # Look back 7 words
        hidden_size=64,           # Larger IRF networks
        combiner_size=64,         # Larger combiner
        dropout=0.2,              # More dropout
        # Training settings
        batch_size=2048,          # Larger batches
        n_epochs=100,             # More epochs
        learning_rate=0.005,      # Lower learning rate
        weight_decay=1e-3,        # More regularization
        early_stopping_patience=20,
        # Data preprocessing
        outlier_percentile=98.0,  # More aggressive outlier handling
        use_log_transform=True,
        test_size=0.2,
        # Other
        device='cuda',
        visualize=True
    )
    
    print(f"\nAdvanced analysis complete!")
    print(f"Final validation R²: {results_advanced['results']['r2']:.4f}")
    
    # =========================================================================
    # WORKING WITH THE RESULTS
    # =========================================================================
    
    # Extract the trained model
    model = results_advanced['model']
    
    # Get normalization parameters for denormalizing predictions
    norm_params = results_advanced['normalization_params']
    
    # Access training history
    history = results_advanced['history']
    print(f"\nBest epoch: {np.argmin(history['val_loss'])}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    
    # Get IRF curves for specific predictors
    print("\nExtracting IRF curves...")
    
    for predictor_idx, predictor_name in enumerate(['surprisal', 'wordlen', 'freq']):
        curves = model.get_irf_curves(
            predictor_idx=predictor_idx,
            t_range=(0, 7),  # Match history_length
            n_points=100,
            predictor_values=[-2, -1, 0, 1, 2]
        )
        
        print(f"\nIRF curves for {predictor_name}:")
        for name, curve_data in curves.items():
            # Find peak effect
            peak_idx = np.argmax(np.abs(curve_data['irf']))
            peak_time = curve_data['time'][peak_idx]
            peak_value = curve_data['irf'][peak_idx]
            print(f"  {name}: peak effect = {peak_value:.3f} at lag = {peak_time:.2f} words")
    
    # =========================================================================
    # MAKING PREDICTIONS ON NEW DATA
    # =========================================================================
    
    print("\n" + "="*80)
    print("MAKING PREDICTIONS ON NEW DATA")
    print("="*80)
    
    # Note: For real prediction on new data, you would need to:
    # 1. Prepare the new data using the same preprocessing pipeline
    # 2. Create windows using the same history_length
    # 3. Normalize using the saved normalization parameters
    # 4. Run the model forward pass
    # 5. Denormalize the predictions
    
    # Note: In a real application, you would need to track validation indices
    # For this example, we'll just show how to work with the results
    
    # The predictions are already computed in results
    predictions_normalized = results_advanced['results']['predictions']
    
    # Denormalize predictions
    predictions_original_scale = (predictions_normalized * norm_params['response_std'] + 
                                 norm_params['response_mean'])
    
    # If log transform was used, reverse it
    if norm_params['use_log_transform']:
        predictions_original_scale = np.expm1(predictions_original_scale)
    
    print(f"Prediction statistics (original scale):")
    print(f"  Mean: {predictions_original_scale.mean():.1f} ms")
    print(f"  Std: {predictions_original_scale.std():.1f} ms")
    print(f"  Range: [{predictions_original_scale.min():.1f}, "
          f"{predictions_original_scale.max():.1f}] ms")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()