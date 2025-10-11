# TS2Vec Training Notebook Fix Documentation

## Issue Summary

The `notebooks/ts2vec-training.ipynb` notebook was failing with a `NameError: name 'trainer' is not defined` error when attempting to evaluate the trained model. The root cause was an API mismatch between the notebook's expected interface and the actual `Trainer` class implementation.

## Problem Details

### Original Error
```python
# Load best model
trainer.load_checkpoint('/kaggle/working/checkpoints/best_model.pt')

# Predictions
ts2vec_trainer.model.eval()  # This line showed error in problem statement
...

NameError: name 'trainer' is not defined
```

### Root Cause Analysis

The notebook (Section 5: Training) used this code:
```python
trainer = Trainer(
    model=model,
    device=device,
    checkpoint_dir='/kaggle/working/checkpoints'  # ❌ Parameter didn't exist
)

history = trainer.train(  # ❌ Method didn't exist
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    X_val=X_val_tensor,
    y_val=y_val_tensor,
    batch_size=64,
    epochs=200,
    learning_rate=0.0005,
    patience=30,
    min_delta=0.5
)
```

But the original `Trainer` class:
- ❌ Did NOT accept `checkpoint_dir` parameter
- ❌ Did NOT have a `train()` method (only `fit()`)
- ❌ Required DataLoader objects instead of tensors
- ❌ Did NOT track `best_val_mae` attribute

## Solution Implemented

### 1. Enhanced Trainer Constructor

**Added Parameters:**
- `checkpoint_dir` (str): Directory to save model checkpoints (default: 'checkpoints')

**Added Attributes:**
- `best_val_mae` (float): Track best validation MAE for reporting

```python
def __init__(self, model, device='cuda', learning_rate=0.001, 
             weight_decay=0.01, checkpoint_dir='checkpoints'):
    # ... existing code ...
    self.checkpoint_dir = Path(checkpoint_dir)
    self.best_val_mae = float('inf')
```

### 2. Added train() Method

Created a new convenience method that matches the notebook's expected API:

```python
def train(self, X_train, y_train, X_val, y_val, batch_size=32, 
          epochs=100, learning_rate=None, patience=15, min_delta=0.0):
    """
    Train model with tensors directly (convenience method for notebooks).
    
    Internally creates DataLoaders and delegates to fit() method.
    """
```

**Features:**
- ✅ Accepts PyTorch tensors directly
- ✅ Creates DataLoaders internally
- ✅ Supports all notebook parameters
- ✅ Uses `checkpoint_dir` from constructor
- ✅ Delegates to existing `fit()` method

### 3. Enhanced fit() Method

- Now tracks `best_val_mae` alongside `best_val_loss`
- Updates both metrics when saving best model

### 4. Fixed PyTorch Compatibility

Removed `verbose=True` parameter from `ReduceLROnPlateau` scheduler:
```python
# Old (incompatible with PyTorch 2.x)
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# New (compatible)
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=5
)
```

## Files Modified

- `src/training/trainer.py`: Enhanced with notebook-compatible API

## Files Added

- `BUGFIX_SUMMARY.md`: Brief summary of the fix
- `FIX_DOCUMENTATION.md`: This comprehensive documentation
- `test_notebook_compatibility.py`: Automated test suite
- `.gitignore`: Updated to exclude training artifacts

## Testing

### Test Suite Created

The `test_notebook_compatibility.py` script validates:

1. ✅ All required modules can be imported
2. ✅ TS2Vec submodule is available
3. ✅ Trainer accepts `checkpoint_dir` parameter
4. ✅ Trainer has `best_val_mae` attribute
5. ✅ Trainer has `train()` method
6. ✅ Complete training workflow works
7. ✅ Checkpoint loading and inference work

**Run the test:**
```bash
python test_notebook_compatibility.py
```

### Manual Validation

The exact notebook code was tested and confirmed working:

```python
# Section 5: Train Model
trainer = Trainer(
    model=model,
    device=device,
    checkpoint_dir='/kaggle/working/checkpoints'
)

history = trainer.train(
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    X_val=X_val_tensor,
    y_val=y_val_tensor,
    batch_size=64,
    epochs=200,
    learning_rate=0.0005,
    patience=30,
    min_delta=0.5
)

print(f"Best epoch: {trainer.best_epoch}")
print(f"Best validation MAE: {trainer.best_val_mae:.2f} mg/dL")

# Section 6: Evaluate Model
trainer.load_checkpoint('/kaggle/working/checkpoints/best_model.pt')

model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_pred = model(X_test_tensor).cpu().numpy().squeeze()

metrics = compute_all_metrics(y_test, y_pred)
print_metrics(metrics)
```

## Backward Compatibility

✅ **All changes are backward compatible**

Existing code using the `fit()` method with DataLoaders will continue to work:

```python
# Old API still works
trainer = Trainer(model, device='cuda', learning_rate=0.001)
trainer.fit(train_loader, val_loader, epochs=100, save_dir='checkpoints')
```

## Usage Examples

### Notebook-Style API (New)
```python
# Create tensors
X_train = torch.FloatTensor(train_data)
y_train = torch.FloatTensor(train_labels).unsqueeze(1)

# Initialize trainer
trainer = Trainer(
    model=model,
    device='cuda',
    checkpoint_dir='./checkpoints'
)

# Train directly with tensors
history = trainer.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    batch_size=64,
    epochs=100,
    learning_rate=0.001,
    patience=15
)

# Access metrics
print(f"Best epoch: {trainer.best_epoch}")
print(f"Best MAE: {trainer.best_val_mae:.2f}")
```

### DataLoader API (Original)
```python
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize trainer
trainer = Trainer(model, device='cuda')

# Train with DataLoaders
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_dir='checkpoints'
)
```

## Verification Steps

To verify the fix works in your environment:

1. **Clone and setup:**
```bash
git clone --recurse-submodules https://github.com/omar-A-hassan/3x1-PPG.git
cd 3x1-PPG
pip install -r requirements.txt
```

2. **Run compatibility test:**
```bash
python test_notebook_compatibility.py
```

3. **Expected output:**
```
============================================================
Notebook Compatibility Test Suite
============================================================
Testing imports...
✅ PASS: All imports successful

Testing Trainer API...
✅ PASS: Trainer API compatible with notebook

Testing training workflow...
✅ PASS: Complete training workflow successful

============================================================
Test Results Summary
============================================================
✅ PASS: Imports
✅ PASS: Trainer API
✅ PASS: Training Workflow
============================================================

🎉 All tests passed! Notebook should work correctly.
```

## Notes

- The fix maintains the original functionality while adding notebook-friendly convenience methods
- Training artifacts (checkpoints, *.pt files) are now excluded via `.gitignore`
- The solution follows the principle of minimal changes - only the Trainer class was modified
- No changes were needed to the notebook itself

## Support

If you encounter any issues:
1. Ensure ts2vec submodule is initialized: `git submodule update --init --recursive`
2. Install all dependencies: `pip install -r requirements.txt`
3. Run the test suite: `python test_notebook_compatibility.py`
4. Check that TS2VEC_AVAILABLE is True: `python -c "from src.models import TS2VEC_AVAILABLE; print(TS2VEC_AVAILABLE)"`
