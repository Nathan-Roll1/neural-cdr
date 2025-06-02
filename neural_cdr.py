"""
Neural Continuous-Time Deconvolutional Regression (Neural CDR)
=============================================================

Core implementation of Neural CDR with learned impulse response functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import gc
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from utils_windows import NUMBA_AVAILABLE, create_windows_numba, create_windows_vectorized
from visualize import visualize_results

warnings.filterwarnings('ignore')


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class CDRDataset(Dataset):
    """PyTorch Dataset for CDR data"""
    def __init__(self, response, predictors, t_delta, mask):
        self.response = response
        self.predictors = predictors
        self.t_delta = t_delta
        self.mask = mask

    def __len__(self):
        return len(self.response)

    def __getitem__(self, idx):
        return {
            'response': self.response[idx],
            'predictors': self.predictors[idx],
            't_delta': self.t_delta[idx],
            'mask': self.mask[idx]
        }


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class ImprovedNeuralIRF(nn.Module):
    """Neural network for learning impulse response functions"""
    def __init__(self, hidden_size=32, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Better weight initialization with controlled scale
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, t_delta, predictor_values):
        # Time encoding with safety checks
        t_delta_safe = torch.abs(t_delta) + 1e-6
        t_delta_norm = torch.log1p(t_delta_safe) / 3.0
        t_delta_norm = torch.clamp(t_delta_norm, -10, 10)

        # Stack inputs
        inputs = torch.stack([t_delta_norm, predictor_values], dim=-1)

        # Forward pass
        x = self.fc1(inputs)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)

        # Final safety clamp
        x = torch.clamp(x, -10, 10)

        return x.squeeze(-1)


class ImprovedBlockLinearCDR(nn.Module):
    """Main Neural CDR model with block-linear architecture"""

    def __init__(self,
                 n_predictors=1,
                 history_length=5,
                 hidden_size=32,
                 combiner_size=32,
                 dropout=0.1):
        super().__init__()

        self.n_predictors = n_predictors
        self.history_length = history_length

        # IRF for each predictor
        self.irf_blocks = nn.ModuleList([
            ImprovedNeuralIRF(hidden_size, dropout) for _ in range(n_predictors)
        ])

        # Learnable parameters
        self.intercept = nn.Parameter(torch.zeros(1))
        self.rate = nn.Parameter(torch.ones(1) * 0.01)

        # Combination network
        self.combiner = nn.Sequential(
            nn.Linear(n_predictors + 2, combiner_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combiner_size, combiner_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combiner_size // 2, 1)
        )

        # Initialize combiner
        for m in self.combiner.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, batch):
        predictors = batch['predictors']
        t_delta = batch['t_delta']
        mask = batch['mask']

        batch_size = t_delta.shape[0]

        if predictors.dim() == 2:
            predictors = predictors.unsqueeze(-1)

        # Compute IRF values
        irf_outputs = []

        for p_idx in range(self.n_predictors):
            pred_values = predictors[:, :, p_idx]
            irf_values = self.irf_blocks[p_idx](t_delta, pred_values)

            # Apply mask and sum
            masked_irf = irf_values * pred_values * mask
            convolution = masked_irf.sum(dim=1)

            irf_outputs.append(convolution)

        # Stack convolutions
        all_convolutions = torch.stack(irf_outputs, dim=1)

        # Add baseline effects
        history_effect = self.rate * mask.sum(dim=1, keepdim=True)
        intercept_expanded = self.intercept.expand(batch_size, 1)

        # Combine features
        combined = torch.cat([
            all_convolutions,
            history_effect,
            intercept_expanded
        ], dim=1)

        # Final prediction
        predictions = self.combiner(combined).squeeze(-1)

        return predictions

    def get_irf_curves(self, predictor_idx=0, t_range=(0, 5), n_points=100,
                      predictor_values=None, device='cuda'):
        """Extract IRF curves for visualization"""
        if predictor_values is None:
            predictor_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

        device = next(self.parameters()).device
        t_values = torch.linspace(t_range[0], t_range[1], n_points).to(device)
        curves = {}

        self.eval()
        with torch.no_grad():
            for pred_val in predictor_values:
                t_delta = t_values.unsqueeze(0)
                pred_tensor = torch.full_like(t_delta, pred_val)

                irf_values = self.irf_blocks[predictor_idx](t_delta, pred_tensor)

                curves[f'predictor={pred_val:.1f}'] = {
                    'time': t_values.cpu().numpy(),
                    'irf': irf_values.squeeze().cpu().numpy()
                }
        self.train()

        return curves


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_reading_data_for_cdr(X, y, predictor_cols, response_col='fdur',
                                history_length=5, future_length=0,
                                device='cuda', outlier_percentile=99.0,
                                use_log_transform=True):
    """
    Prepare reading time data with fast window processing and outlier handling
    
    Args:
        X: DataFrame with predictors
        y: DataFrame with responses
        predictor_cols: List of predictor column names
        response_col: Name of response column
        history_length: Number of past observations to include
        future_length: Number of future observations to include
        device: PyTorch device
        outlier_percentile: Percentile for outlier capping
        use_log_transform: Whether to log-transform response
        
    Returns:
        Dictionary with processed data and normalization parameters
    """
    start_time = time.time()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Validate inputs
    if response_col not in y.columns:
        raise ValueError(f"'{response_col}' column not found in y DataFrame")

    print(f"\nInput data shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    # Check predictor columns
    missing_predictors = [col for col in predictor_cols if col not in X.columns]
    if missing_predictors:
        raise ValueError(f"Missing predictor columns in X: {missing_predictors}")

    # Remove response from X if it exists
    if response_col in X.columns:
        print(f"Warning: '{response_col}' found in X. Removing from X for merge.")
        X = X.drop(columns=[response_col])

    # Find merge columns
    common_cols = list(set(X.columns) & set(y.columns))
    merge_candidates = ['time', 'sentid', 'sentpos', 'word', 'subject', 'item', 'zone']
    merge_cols = [col for col in merge_candidates if col in common_cols]

    if not merge_cols:
        raise ValueError(f"No suitable merge columns found")

    print(f"Merging on columns: {merge_cols}")

    # Subset columns before merge
    X_cols_needed = predictor_cols + merge_cols
    X_cols_needed = list(dict.fromkeys(X_cols_needed))
    X_subset = X[X_cols_needed].copy()

    y_cols_needed = [response_col] + merge_cols
    y_cols_needed = list(dict.fromkeys(y_cols_needed))
    y_subset = y[y_cols_needed].copy()

    # Merge data
    merge_start = time.time()
    data = pd.merge(X_subset, y_subset, on=merge_cols, how='inner')
    print(f"  Merge completed in {time.time() - merge_start:.1f} seconds")

    # Validate response values
    print(f"\n{response_col} statistics before filtering:")
    print(f"  Mean: {data[response_col].mean():.2f}")
    print(f"  Std: {data[response_col].std():.2f}")
    print(f"  Min: {data[response_col].min():.2f}")
    print(f"  Max: {data[response_col].max():.2f}")
    print(f"  % zeros: {(data[response_col] == 0).mean() * 100:.1f}%")

    # Filter invalid response times
    initial_len = len(data)
    data = data[data[response_col] > 0].reset_index(drop=True)
    print(f"\nFiltered out {initial_len - len(data)} rows with {response_col} <= 0")

    if len(data) == 0:
        raise ValueError("No valid data remaining after filtering!")

    # Handle outliers
    print(f"\n*** OUTLIER HANDLING ***")
    original_response = data[response_col].values

    # Show percentiles
    print("Original percentiles:")
    for p in [50, 90, 95, 99, 99.9, 100]:
        print(f"  {p:5.1f}%: {np.percentile(original_response, p):10.1f} ms")

    # Cap outliers
    cap_value = None
    if outlier_percentile < 100:
        cap_value = np.percentile(original_response, outlier_percentile)
        data['response_capped'] = np.clip(data[response_col], 0, cap_value)
        print(f"\nCapping outliers at {outlier_percentile}th percentile ({cap_value:.1f} ms)")
        print(f"  Affected rows: {(data[response_col] > cap_value).sum()} ({(data[response_col] > cap_value).mean()*100:.2f}%)")
    else:
        data['response_capped'] = data[response_col]

    # Apply log transform if requested
    if use_log_transform:
        data['response_transformed'] = np.log1p(data['response_capped'])
        print(f"\nApplied log1p transform")
        response_col_final = 'response_transformed'
    else:
        data['response_transformed'] = data['response_capped']
        response_col_final = 'response_transformed'

    print(f"\nTransformed statistics:")
    print(f"  Mean: {data[response_col_final].mean():.4f}")
    print(f"  Std: {data[response_col_final].std():.4f}")

    # Sort data
    sort_cols = []
    if 'sentid' in data.columns:
        sort_cols.append('sentid')
    if 'time' in data.columns:
        sort_cols.append('time')
    elif 'sentpos' in data.columns:
        sort_cols.append('sentpos')

    if sort_cols:
        print(f"  Sorting by: {sort_cols}")
        data = data.sort_values(sort_cols).reset_index(drop=True)

    # Add time column if missing
    if 'time' not in data.columns:
        print("  Creating time column from row indices")
        data['time'] = np.arange(len(data))

    # Get sentid
    if 'sentid' in data.columns:
        sentids = data['sentid'].values.astype(np.int32)
    else:
        sentids = np.zeros(len(data), dtype=np.int32)

    n_obs = len(data)
    n_predictors = len(predictor_cols)
    window_size = history_length + future_length

    print(f"\nProcessing {n_obs:,} observations")
    print(f"Window size: {window_size}")

    # Prepare data
    times = data['time'].values.astype(np.float32)
    predictor_data = data[predictor_cols].fillna(0).values.astype(np.float32)

    # Check for extreme values in predictors
    print(f"\n  Predictor statistics before normalization:")
    for i, col in enumerate(predictor_cols):
        col_data = predictor_data[:, i]
        print(f"    {col}: mean={col_data.mean():.4f}, std={col_data.std():.4f}, "
              f"range=[{col_data.min():.4f}, {col_data.max():.4f}]")

        # Clip extreme values
        extreme_count = ((col_data < -100) | (col_data > 100)).sum()
        if extreme_count > 0:
            print(f"      WARNING: {extreme_count} extreme values (|x| > 100)")
            predictor_data[:, i] = np.clip(col_data, -100, 100)

    # Normalize predictors
    predictor_means = predictor_data.mean(axis=0)
    predictor_stds = predictor_data.std(axis=0) + 1e-8
    predictor_data = (predictor_data - predictor_means) / predictor_stds

    # Create windows
    print("\nCreating time-lagged windows (optimized)...")
    window_start_time = time.time()

    if NUMBA_AVAILABLE:
        predictors_np, t_delta_np, mask_np = create_windows_numba(
            times, sentids, predictor_data,
            history_length, n_obs, n_predictors
        )
    else:
        predictors_np, t_delta_np, mask_np = create_windows_vectorized(
            times, sentids, predictor_data, history_length
        )

    print(f"  Window creation completed in {time.time() - window_start_time:.1f} seconds")

    # Debug window statistics
    print("\n  Window statistics:")
    print(f"    Predictors shape: {predictors_np.shape}")
    print(f"    Non-zero mask elements: {(mask_np > 0).sum()} / {mask_np.size}")

    # Convert to PyTorch tensors
    predictors = torch.from_numpy(predictors_np).to(device, non_blocking=True)
    t_delta = torch.from_numpy(t_delta_np).to(device, non_blocking=True)
    mask = torch.from_numpy(mask_np).to(device, non_blocking=True)

    # Normalize response
    response_values = data[response_col_final].values.astype(np.float32)
    response_mean = float(response_values.mean())
    response_std = float(response_values.std())

    if response_std < 1e-6:
        print(f"WARNING: Response std is very small ({response_std}), using 1.0")
        response_std = 1.0

    response_normalized = (response_values - response_mean) / response_std

    print(f"\nNormalization check:")
    print(f"  Before: mean={response_mean:.4f}, std={response_std:.4f}")
    print(f"  After: mean={response_normalized.mean():.6f}, std={response_normalized.std():.6f}")

    # Convert to tensor
    response = torch.tensor(response_normalized, dtype=torch.float32, device=device)

    print(f"\nTotal data preparation time: {time.time() - start_time:.1f} seconds")

    # Create metadata
    metadata = pd.DataFrame({
        response_col: data[response_col].values,
        'response_capped': data['response_capped'].values if 'response_capped' in data else data[response_col].values,
        'response_transformed': data[response_col_final].values,
        'response_normalized': response.cpu().numpy()
    })

    # Add normalization parameters
    normalization_params = {
        'response_mean': response_mean,
        'response_std': response_std,
        'predictor_means': predictor_means,
        'predictor_stds': predictor_stds,
        'outlier_percentile': outlier_percentile,
        'use_log_transform': use_log_transform,
        'cap_value': cap_value
    }

    return {
        'response': response,
        'predictors': predictors,
        't_delta': t_delta,
        'mask': mask,
        'metadata': metadata,
        'normalization_params': normalization_params
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def compute_r2(predictions, targets):
    """Compute R² score"""
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    try:
        r2 = r2_score(target_np, pred_np)
    except:
        r2 = float('nan')

    return r2


def train_model(train_data, val_data, model_params, training_params):
    """
    Train Neural CDR model
    
    Args:
        train_data: Dictionary with training data
        val_data: Dictionary with validation data
        model_params: Dictionary with model hyperparameters
        training_params: Dictionary with training hyperparameters
        
    Returns:
        model: Trained model
        history: Training history
    """
    
    device = torch.device(training_params.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    n_predictors = train_data['predictors'].shape[-1]

    # Create model
    model = ImprovedBlockLinearCDR(
        n_predictors=n_predictors,
        history_length=model_params.get('history_length', 5),
        hidden_size=model_params.get('hidden_size', 32),
        combiner_size=model_params.get('combiner_size', 32),
        dropout=model_params.get('dropout', 0.1)
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset = CDRDataset(
        train_data['response'],
        train_data['predictors'],
        train_data['t_delta'],
        train_data['mask']
    )
    val_dataset = CDRDataset(
        val_data['response'],
        val_data['predictors'],
        val_data['t_delta'],
        val_data['mask']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params.get('batch_size', 1024),
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params.get('batch_size', 1024) * 2,
        shuffle=False,
        num_workers=0
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params.get('learning_rate', 1e-2) * 0.1,  # Start with warmup
        weight_decay=training_params.get('weight_decay', 1e-4),
        eps=1e-4
    )

    # Warm-up scheduler
    warmup_steps = len(train_loader) * 2
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=True
    )

    criterion = nn.MSELoss()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    n_epochs = training_params.get('n_epochs', 50)

    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        train_pbar = tqdm(train_loader,
                         desc=f"Epoch {epoch:2d}/{n_epochs-1} [Train]",
                         ncols=100,
                         leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()

            pred = model(batch)
            loss = criterion(pred, batch['response'])

            if torch.isnan(loss):
                print(f"\n  ERROR: NaN loss at epoch {epoch}, batch {batch_idx}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch < 2:
                warmup_scheduler.step()

            train_losses.append(loss.item())
            train_preds.append(pred.detach())
            train_targets.append(batch['response'].detach())

            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        # Calculate training metrics
        avg_train_loss = np.mean(train_losses)
        all_train_preds = torch.cat(train_preds)
        all_train_targets = torch.cat(train_targets)
        train_r2 = compute_r2(all_train_preds, all_train_targets)

        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        val_pbar = tqdm(val_loader,
                       desc=f"Epoch {epoch:2d}/{n_epochs-1} [Val]  ",
                       ncols=100,
                       leave=False)

        with torch.no_grad():
            for batch in val_pbar:
                pred = model(batch)
                loss = criterion(pred, batch['response'])

                val_losses.append(loss.item())
                val_preds.append(pred)
                val_targets.append(batch['response'])

                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

        # Calculate validation metrics
        avg_val_loss = np.mean(val_losses)
        all_val_preds = torch.cat(val_preds)
        all_val_targets = torch.cat(val_targets)
        val_r2 = compute_r2(all_val_preds, all_val_targets)

        # Print summary
        if epoch % 5 == 0 or epoch == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch:2d}/{n_epochs-1} Summary:")
            print(f"  Train - Loss: {avg_train_loss:.4f}, R²: {train_r2:.4f}")
            print(f"  Val   - Loss: {avg_val_loss:.4f}, R²: {val_r2:.4f}")
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

            # Early stopping
            if patience_counter >= training_params.get('early_stopping_patience', 15):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with val_loss: {best_val_loss:.4f}")

    return model, history


# =============================================================================
# MAIN WRAPPER FUNCTION
# =============================================================================

def run_neural_cdr(X, y, 
                  predictor_cols,
                  response_col='fdur',
                  # Model hyperparameters
                  history_length=5,
                  future_length=0,
                  hidden_size=32,
                  combiner_size=32,
                  dropout=0.1,
                  # Training hyperparameters  
                  batch_size=1024,
                  n_epochs=50,
                  learning_rate=1e-2,
                  weight_decay=1e-4,
                  early_stopping_patience=15,
                  # Data preprocessing
                  outlier_percentile=99.0,
                  use_log_transform=True,
                  test_size=0.2,
                  random_state=42,
                  # Other options
                  device='cuda',
                  visualize=True):
    """
    Complete Neural CDR analysis with all hyperparameter options
    
    Args:
        X: DataFrame with predictors
        y: DataFrame with responses
        predictor_cols: List of predictor column names
        response_col: Name of response column (default: 'fdur')
        
        Model hyperparameters:
        history_length: Number of previous observations to include (default: 5)
        future_length: Number of future observations to include (default: 0)
        hidden_size: Hidden layer size for IRF networks (default: 32)
        combiner_size: Hidden layer size for combiner network (default: 32)
        dropout: Dropout rate (default: 0.1)
        
        Training hyperparameters:
        batch_size: Batch size for training (default: 1024)
        n_epochs: Number of training epochs (default: 50)
        learning_rate: Learning rate (default: 0.01)
        weight_decay: Weight decay for AdamW (default: 1e-4)
        early_stopping_patience: Epochs to wait before early stopping (default: 15)
        
        Data preprocessing:
        outlier_percentile: Percentile for outlier capping (default: 99.0)
        use_log_transform: Whether to log-transform response (default: True)
        test_size: Fraction of data for validation (default: 0.2)
        random_state: Random seed (default: 42)
        
        Other options:
        device: PyTorch device (default: 'cuda')
        visualize: Whether to create visualizations (default: True)
        
    Returns:
        Dictionary containing:
        - model: Trained Neural CDR model
        - data: Processed data dictionary
        - history: Training history
        - results: Analysis results (if visualize=True)
        - normalization_params: Parameters for denormalizing predictions
    """
    
    print("=" * 80)
    print("NEURAL CONTINUOUS-TIME DECONVOLUTIONAL REGRESSION")
    print("=" * 80)
    print(f"\nUsing acceleration: {'Numba' if NUMBA_AVAILABLE else 'NumPy vectorization'}")
    
    # Set CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        gc.collect()
    
    # Prepare data
    total_start = time.time()
    
    cdr_data = prepare_reading_data_for_cdr(
        X, y,
        predictor_cols=predictor_cols,
        response_col=response_col,
        history_length=history_length,
        future_length=future_length,
        outlier_percentile=outlier_percentile,
        use_log_transform=use_log_transform,
        device=device
    )
    
    print(f"\nTotal valid observations: {len(cdr_data['response']):,}")
    
    # Split data
    n_obs = len(cdr_data['response'])
    indices = np.arange(n_obs)
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    # Create train/val splits
    train_data = {
        'response': cdr_data['response'][train_idx],
        'predictors': cdr_data['predictors'][train_idx],
        't_delta': cdr_data['t_delta'][train_idx],
        'mask': cdr_data['mask'][train_idx]
    }
    
    val_data = {
        'response': cdr_data['response'][val_idx],
        'predictors': cdr_data['predictors'][val_idx],
        't_delta': cdr_data['t_delta'][val_idx],
        'mask': cdr_data['mask'][val_idx]
    }
    
    print(f"\nTrain size: {len(train_data['response']):,}")
    print(f"Validation size: {len(val_data['response']):,}")
    
    # Model parameters
    model_params = {
        'history_length': history_length,
        'hidden_size': hidden_size,
        'combiner_size': combiner_size,
        'dropout': dropout
    }
    
    # Training parameters
    training_params = {
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'early_stopping_patience': early_stopping_patience,
        'device': device
    }
    
    # Train model
    model, history = train_model(
        train_data, val_data,
        model_params, training_params
    )
    
    # Visualize results
    results = None
    if visualize:
        results = visualize_results(
            model, history, val_data,
            cdr_data['normalization_params'],
            CDRDataset
        )
    
    print(f"\nTotal analysis time: {time.time() - total_start:.1f} seconds")
    
    return {
        'model': model,
        'data': cdr_data,
        'history': history,
        'results': results,
        'normalization_params': cdr_data['normalization_params']
    }