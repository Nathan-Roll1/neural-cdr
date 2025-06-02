# Getting Started with Neural CDR

## Quick Example

```python
from neural_cdr import run_neural_cdr
import pandas as pd

# Load your data
X = pd.read_csv('predictors.csv')  # Must contain predictor columns
y = pd.read_csv('responses.csv')   # Must contain response column (e.g., 'fdur')

# Run basic analysis
results = run_neural_cdr(
    X, y,
    predictor_cols=['surprisal', 'wordlen', 'freq'],
    response_col='fdur'
)

# Access results
print(f"Validation R²: {results['results']['r2']:.4f}")
model = results['model']
```

## Complete Parameter Reference

```python
results = run_neural_cdr(
    X, y,
    predictor_cols=['surprisal', 'wordlen'],    # Required: list of predictor columns
    response_col='fdur',                        # Response column name
    
    # Model Architecture
    history_length=5,        # Number of previous words to consider
    future_length=0,         # Number of future words (for preview effects)
    hidden_size=32,          # IRF network hidden layer size
    combiner_size=32,        # Combiner network hidden layer size
    dropout=0.1,             # Dropout rate
    
    # Training Parameters
    batch_size=1024,         # Batch size
    n_epochs=50,             # Maximum epochs
    learning_rate=0.01,      # Initial learning rate
    weight_decay=1e-4,       # L2 regularization
    early_stopping_patience=15,  # Epochs before early stopping
    
    # Data Preprocessing
    outlier_percentile=99.0, # Cap outliers at this percentile
    use_log_transform=True,  # Log-transform response variable
    test_size=0.2,           # Validation set fraction
    random_state=42,         # Random seed
    
    # Other Options
    device='cuda',           # 'cuda' or 'cpu'
    visualize=True          # Generate diagnostic plots
)
```

## Data Requirements

Your data must have:
- **X DataFrame**: Contains predictor columns and merge keys
- **Y DataFrame**: Contains response column and merge keys
- **Common columns** for merging (e.g., 'sentid', 'sentpos', 'time')
- **Positive response values** (zero/negative values are filtered)

Example data structure:
```
X columns: ['sentid', 'sentpos', 'word', 'surprisal', 'wordlen', 'freq']
y columns: ['sentid', 'sentpos', 'fdur']
```

## Understanding the Output

The function returns a dictionary with:
- `model`: Trained PyTorch model
- `data`: Processed tensors and metadata
- `history`: Training/validation losses and R² scores
- `results`: Predictions and performance metrics (if visualize=True)
- `normalization_params`: Parameters for denormalizing predictions

## Common Use Cases

### 1. Basic surprisal analysis
```python
results = run_neural_cdr(X, y, predictor_cols=['surprisal'])
```

### 2. Multiple predictors with custom architecture
```python
results = run_neural_cdr(
    X, y,
    predictor_cols=['surprisal', 'entropy', 'wordlen'],
    history_length=10,      # Longer history
    hidden_size=64,         # Larger networks
    batch_size=4096        # Larger batches for GPU
)
```

### 3. Self-paced reading data
```python
results = run_neural_cdr(
    X, y,
    predictor_cols=['surprisal'],
    response_col='rt',      # Different response column
    outlier_percentile=95,  # More aggressive outlier removal
    use_log_transform=False # SPR often doesn't need log transform
)
```

### 4. Extracting IRF curves
```python
# After training
model = results['model']
curves = model.get_irf_curves(
    predictor_idx=0,              # First predictor
    t_range=(0, 5),              # Time range in words
    predictor_values=[-2, 0, 2]  # Predictor values to plot
)
```

## Tips for Best Results

1. **Start simple**: Begin with one predictor and default parameters
2. **Check your data**: Ensure proper time ordering within sentences
3. **Monitor overfitting**: Watch the loss ratio plot
4. **Tune carefully**: 
   - Increase `history_length` for longer dependencies
   - Increase `hidden_size` for more complex IRF shapes
   - Adjust `outlier_percentile` based on your data distribution
5. **Use GPU**: Set `device='cuda'` for 10-100x speedup