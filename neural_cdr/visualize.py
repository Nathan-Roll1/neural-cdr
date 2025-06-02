"""
Visualization Functions for Neural CDR
======================================

Comprehensive visualization and diagnostic plotting functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy import stats


def visualize_results(model, history, val_data, normalization_params, dataset_class):
    """
    Comprehensive visualization of Neural CDR results
    
    Args:
        model: Trained Neural CDR model
        history: Training history dictionary
        val_data: Validation data dictionary
        normalization_params: Dictionary with normalization parameters
        dataset_class: CDRDataset class for creating data loaders
        
    Returns:
        Dictionary with analysis results
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_loss'], label='Train', linewidth=2, color='blue')
    ax1.plot(history['val_loss'], label='Validation', linewidth=2, color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. R² curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['train_r2'], label='Train', linewidth=2, color='green')
    ax2.plot(history['val_r2'], label='Validation', linewidth=2, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.set_title('R² Score Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.0)

    # 3. Learning rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['learning_rates'], linewidth=2, color='purple')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. IRF curves
    ax4 = fig.add_subplot(gs[1, :2])
    curves = model.get_irf_curves(
        predictor_idx=0,
        t_range=(0, 5),
        n_points=100,
        predictor_values=[-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    colors = plt.cm.viridis(np.linspace(0, 1, len(curves)))
    for i, (name, curve_data) in enumerate(curves.items()):
        ax4.plot(curve_data['time'], curve_data['irf'],
                color=colors[i], label=name, linewidth=2)

    ax4.set_xlabel('Time Lag (words)')
    ax4.set_ylabel('IRF Value')
    ax4.set_title('Impulse Response Functions by Predictor Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # 5. Predictions vs Actual
    ax5 = fig.add_subplot(gs[1, 2])

    # Get predictions
    model.eval()
    val_dataset = dataset_class(
        val_data['response'],
        val_data['predictors'],
        val_data['t_delta'],
        val_data['mask']
    )
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch)
            all_preds.append(pred.cpu())
            all_targets.append(batch['response'].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Sample for visualization
    idx = np.random.choice(len(all_preds), min(5000, len(all_preds)), replace=False)

    ax5.scatter(all_targets[idx], all_preds[idx], alpha=0.3, s=10)
    ax5.plot([all_targets.min(), all_targets.max()],
             [all_targets.min(), all_targets.max()],
             'r--', linewidth=2)
    ax5.set_xlabel('Actual (normalized)')
    ax5.set_ylabel('Predicted (normalized)')
    ax5.set_title('Predictions vs Actual (Validation)')
    ax5.grid(True, alpha=0.3)

    # 6. Residual distribution
    ax6 = fig.add_subplot(gs[2, 0])
    residuals = all_targets - all_preds
    ax6.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax6.set_xlabel('Residuals')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Residual Distribution (std={residuals.std():.3f})')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. IRF heatmap
    ax7 = fig.add_subplot(gs[2, 1])

    t_lags = np.arange(0, 5, 0.1)
    pred_vals = np.linspace(-2, 2, 20)
    heatmap_data = np.zeros((len(pred_vals), len(t_lags)))

    model.eval()
    with torch.no_grad():
        for i, pred_val in enumerate(pred_vals):
            t_delta = torch.tensor(t_lags, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
            pred_tensor = torch.full_like(t_delta, pred_val)
            irf_values = model.irf_blocks[0](t_delta, pred_tensor)
            heatmap_data[i, :] = irf_values.squeeze().cpu().numpy()

    im = ax7.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                    extent=[0, 5, -2, 2], origin='lower')
    ax7.set_xlabel('Time Lag (words)')
    ax7.set_ylabel('Predictor Value (normalized)')
    ax7.set_title('IRF Heatmap')
    plt.colorbar(im, ax=ax7, label='IRF Value')

    # 8. Performance summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    final_r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    if normalization_params['use_log_transform']:
        unit_str = "log(ms)"
    else:
        unit_str = "ms"

    rmse_original = rmse * normalization_params['response_std']
    mae_original = mae * normalization_params['response_std']

    summary_text = f"""Final Performance Summary:

    Validation Metrics:
      R²: {final_r2:.4f}
      RMSE: {rmse:.4f} (normalized)
            {rmse_original:.4f} {unit_str}
      MAE: {mae:.4f} (normalized)
           {mae_original:.4f} {unit_str}

    Model Size:
      Parameters: {sum(p.numel() for p in model.parameters()):,}

    Training:
      Best epoch: {np.argmin(history['val_loss'])}
      Final LR: {history['learning_rates'][-1]:.2e}

    Data Processing:
      Outlier percentile: {normalization_params['outlier_percentile']}%
      Log transform: {normalization_params['use_log_transform']}
    """

    ax8.text(0.05, 0.5, summary_text, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 9. Loss ratio
    ax9 = fig.add_subplot(gs[3, 0])
    loss_ratio = np.array(history['val_loss']) / np.array(history['train_loss'])
    ax9.plot(loss_ratio, linewidth=2, color='darkred')
    ax9.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Val Loss / Train Loss')
    ax9.set_title('Loss Ratio (Overfitting Indicator)')
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim(0, max(2, np.percentile(loss_ratio, 95)))

    # 10. Time lag importance
    ax10 = fig.add_subplot(gs[3, 1])

    # Calculate average absolute IRF value at each time lag
    t_lags = np.arange(0, 5, 0.1)
    avg_importance = np.zeros(len(t_lags))

    with torch.no_grad():
        for pred_val in np.linspace(-2, 2, 20):
            t_delta = torch.tensor(t_lags, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
            pred_tensor = torch.full_like(t_delta, pred_val)
            irf_values = model.irf_blocks[0](t_delta, pred_tensor)
            avg_importance += np.abs(irf_values.squeeze().cpu().numpy())

    avg_importance /= 20

    ax10.plot(t_lags, avg_importance, linewidth=2, color='darkgreen')
    ax10.fill_between(t_lags, 0, avg_importance, alpha=0.3, color='green')
    ax10.set_xlabel('Time Lag (words)')
    ax10.set_ylabel('Average |IRF|')
    ax10.set_title('Temporal Importance Profile')
    ax10.grid(True, alpha=0.3)

    # 11. Q-Q plot
    ax11 = fig.add_subplot(gs[3, 2])

    # Theoretical quantiles
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    empirical_quantiles = np.sort(residuals)

    ax11.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)
    ax11.plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
              [theoretical_quantiles.min(), theoretical_quantiles.max()],
              'r--', linewidth=2)
    ax11.set_xlabel('Theoretical Quantiles')
    ax11.set_ylabel('Sample Quantiles')
    ax11.set_title('Q-Q Plot (Normality Check)')
    ax11.grid(True, alpha=0.3)

    plt.suptitle('Complete Neural CDR Analysis Results', fontsize=16)
    plt.tight_layout()
    plt.show()

    return {
        'predictions': all_preds,
        'targets': all_targets,
        'residuals': residuals,
        'r2': final_r2,
        'rmse': rmse,
        'mae': mae
    }