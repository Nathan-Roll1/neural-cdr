"""
Example Neural CDR Analysis
===========================

This script demonstrates how to use Neural CDR on reading time data.
UPDATED: Shows how to use the prevent_leakage option for unbiased validation.
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
    # COMPARISON: WITH AND WITHOUT LEAKAGE PREVENTION
    # =========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS 1: WITH POTENTIAL LEAKAGE (OLD METHOD)")
    print("="*80)
    
    results_with_leakage = run_neural_cdr(
        X, y,
        predictor_cols=['surprisal'],
        response_col='fdur',
        n_epochs=30,  # Fewer epochs for demo
        prevent_leakage=False,  # OLD METHOD
        visualize=False  # Skip visualization for comparison
    )
    
    print(f"\nResults WITH potential leakage:")
    print(f"  Final validation R²: {results_with_leakage['history']['val_r2'][-1]:.4f}")
    print(f"  Best validation R²: {max(results_with_leakage['history']['val_r2']):.4f}")
    
    # =========================================================================
    # ANALYSIS 2: WITHOUT LEAKAGE (RECOMMENDED)
    # =========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS 2: WITHOUT LEAKAGE (RECOMMENDED METHOD)")
    print("="*80)
    
    results_no_leakage = run_neural_cdr(
        X, y,
        predictor_cols=['surprisal'],
        response_col='fdur',
        n_epochs=30,  # Fewer epochs for demo
        prevent_leakage=True,  # NEW METHOD (default)
        split_by='sentence',   # Split by sentences
        visualize=True
    )
    
    print(f"\nResults WITHOUT leakage:")
    print(f"  Final validation R²: {results_no_leakage['history']['val_r2'][-1]:.4f}")
    print(f"  Best validation R²: {max(results_no_leakage['history']['val_r2']):.4f}")
    
    # =========================================================================
    # COMPARE RESULTS
    # =========================================================================
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    r2_with_leakage = max(results_with_leakage['history']['val_r2'])
    r2_no_leakage = max(results_no_leakage['history']['val_r2'])
    
    inflation = (r2_with_leakage - r2_no_leakage) / r2_no_leakage * 100
    
    print(f"\nBest R² with potential leakage: {r2_with_leakage:.4f}")
    print(f"Best R² without leakage: {r2_no_leakage:.4f}")
    print(f"Estimated inflation: {inflation:.1f}%")
    
    if inflation > 10:
        print("\n⚠️  Significant inflation detected! Always use prevent_leakage=True")
    else:
        print("\n✓  Minimal inflation in this dataset")
    
    # =========================================================================
    # ADVANCED USAGE WITH MULTIPLE PREDICTORS
    # =========================================================================
    
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS WITH MULTIPLE PREDICTORS")
    print("="*80)
    
    results_advanced = run_neural_cdr(
        X, y,
        predictor_cols=['surprisal', 'wordlen', 'freq'],
        response_col='fdur',
        # Model architecture
        history_length=7,
        hidden_size=64,
        combiner_size=64,
        dropout=0.2,
        # Training settings
        batch_size=2048,
        n_epochs=50,
        learning_rate=0.005,
        weight_decay=1e-3,
        early_stopping_patience=20,
        # Data settings
        outlier_percentile=98.0,
        use_log_transform=True,
        test_size=0.2,
        # IMPORTANT: Always use these for real analyses
        prevent_leakage=True,
        split_by='sentence',
        # Other
        device='cuda',
        visualize=True
    )
    
    print(f"\nAdvanced analysis complete!")
    print(f"Final validation R² (no leakage): {results_advanced['results']['r2']:.4f}")
    
    # =========================================================================
    # MAKING PREDICTIONS ON NEW DATA
    # =========================================================================
    
    print("\n" + "="*80)
    print("USING THE MODEL FOR PREDICTIONS")
    print("="*80)
    
    # When using prevent_leakage=True, normalization parameters are computed
    # only on training data, which is what you want for deployment
    
    model = results_advanced['model']
    norm_params = results_advanced['normalization_params']
    
    print("\nNormalization parameters (computed on training data only):")
    print(f"  Response mean: {norm_params['response_mean']:.2f}")
    print(f"  Response std: {norm_params['response_std']:.2f}")
    print(f"  Outlier cap: {norm_params['cap_value']:.2f}")
    
    # For deployment, you would:
    # 1. Save these normalization parameters
    # 2. Apply them to any new data
    # 3. Create windows for the new data
    # 4. Run predictions
    
    print("\nAnalysis complete!")
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("1. Always use prevent_leakage=True for unbiased validation")
    print("2. Split by sentences when possible (split_by='sentence')")
    print("3. The reported R² is now trustworthy for publication")
    print("4. Normalization parameters are computed on training data only")


if __name__ == "__main__":
    main()