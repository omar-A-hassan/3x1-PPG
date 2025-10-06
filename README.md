# Non-Invasive Blood Glucose Prediction from PPG Signals

Deep learning system for predicting blood glucose levels from Photoplethysmography (PPG) waveforms using a CNN-GRU hybrid architecture.

## Overview

This project implements a state-of-the-art deep learning model for non-invasive glucose monitoring using PPG signals, based on recent research achieving **MAE of 2.96 mg/dL** and **R² of 0.97**.

### Key Features

- **CNN-GRU Hybrid Architecture**: Combines CNN spatial feature extraction with GRU temporal modeling
- **Comprehensive Preprocessing**: 1-second segmentation, bandpass filtering, peak detection, template matching
- **Clinical Metrics**: RMSE, MAE, MAPE, R², and Clarke Error Grid Analysis
- **Patient-Level Splitting**: Prevents data leakage with proper cross-patient validation
- **Kaggle-Ready**: Complete training pipeline optimized for Kaggle GPU environments

### Performance Goals

Based on "Non-Invasive Glucose Level Monitoring from PPG using a Hybrid CNN-GRU Deep Learning Network" (2024):

| Metric | Target |
|--------|--------|
| MAE | 2.96 mg/dL |
| MAPE | 2.40% |
| R² | 0.97 |
| RMSE | 3.94 mg/dL |
| Clarke Zone A+B | >99% |

## Architecture

### Model Pipeline

```
PPG Signal (100 samples @ 100Hz)
    ↓
CNN Feature Extraction (64→128→256 filters)
    ↓
GRU Temporal Modeling (128→64 hidden units)
    ↓
Dense Layers (32→1)
    ↓
Glucose Prediction (mg/dL)
```

### Model Details

- **Input**: 1-second PPG segments (100 samples at 100Hz)
- **CNN Module**: 3 convolutional layers with BatchNorm and MaxPooling
- **GRU Module**: 2 GRU layers for temporal dependency modeling
- **Output**: Single glucose value (mg/dL)
- **Parameters**: ~312K trainable parameters

## Project Structure

```
3x1-PPG/
├── src/
│   ├── models/
│   │   ├── cnn_module.py          # CNN feature extractor
│   │   ├── gru_module.py          # GRU temporal model
│   │   └── cnn_gru.py             # Complete hybrid model
│   ├── preprocessing/
│   │   └── ppg_preprocessor.py    # Signal preprocessing pipeline
│   ├── training/
│   │   └── trainer.py             # Training loop with early stopping
│   ├── evaluation/
│   │   └── metrics.py             # RMSE, MAE, MAPE, Clarke Error Grid
│   └── utils/
├── notebooks/
│   ├── glucose_prediction_model.ipynb  # Baseline polynomial regression
│   └── kaggle_training.ipynb           # End-to-end CNN-GRU training
├── data/
│   └── ppg_glucose_data.csv       # Sample dataset (23 patients)
└── requirements.txt

```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/3x1-PPG.git
cd 3x1-PPG

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, SciPy
- NeuroKit2 (for PPG peak detection)
- Matplotlib, Seaborn (for visualization)
- scikit-learn (for metrics)

## Quick Start

### 1. Preprocess PPG Signals

```python
from src.preprocessing import PPGPreprocessor

preprocessor = PPGPreprocessor(
    sampling_rate=100,
    segment_length=1.0,  # 1-second segments
    lowcut=0.5,           # Hz
    highcut=8.0           # Hz
)

# Preprocess raw PPG signal
segments = preprocessor.preprocess(ppg_signal, apply_template_matching=True)
```

### 2. Build and Train Model

```python
from src.models import build_cnn_gru
from src.training import train_model

# Build model
model = build_cnn_gru(input_length=100, dropout=0.3)

# Train model
trainer = train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=100,
    device='cuda'
)
```

### 3. Evaluate Performance

```python
from src.evaluation import compute_all_metrics, print_metrics, clarke_error_grid_analysis

# Compute metrics
metrics = compute_all_metrics(y_true, y_pred)
print_metrics(metrics)

# Generate Clarke Error Grid
clarke_error_grid_analysis(y_true, y_pred, save_path='clarke_grid.png')
```

## Datasets

### VitalDB (Recommended)
- **Patients**: 6,388 surgical patients
- **PPG Sampling**: 500 Hz
- **Access**: https://vitaldb.net
- **Size**: Large-scale, diverse patient population

### MUST Dataset
- **Patients**: 67 subjects
- **PPG Sampling**: 2175 Hz
- **Access**: https://data.mendeley.com/datasets/37pm7jk7jn/3
- **Size**: Smaller, normal physiological state

## Training on Kaggle

1. Upload project to Kaggle dataset
2. Create new Kaggle notebook
3. Add dataset to notebook
4. Run `kaggle_training.ipynb`
5. Enable GPU accelerator (P100 or T4)
6. Download trained model weights

See [notebooks/kaggle_training.ipynb](notebooks/kaggle_training.ipynb) for complete training pipeline.

## Model Checkpoints

Trained models are saved with:
- Model state dictionary
- Optimizer state
- Training history
- Validation metrics
- Preprocessor configuration

Load model for inference:

```python
import torch
from src.models import build_cnn_gru

# Load checkpoint
checkpoint = torch.load('cnn_gru_glucose_model.pth')

# Build and load model
model = build_cnn_gru(input_length=100, dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make prediction
glucose_prediction = model(ppg_segment)
```

## Data Splitting Strategy

**Critical**: Always split by patient ID to prevent data leakage:

```python
from sklearn.model_selection import train_test_split

unique_patients = df['patient_id'].unique()
train_patients, test_patients = train_test_split(
    unique_patients,
    test_size=0.2,
    random_state=42
)

X_train = X[df['patient_id'].isin(train_patients)]
X_test = X[df['patient_id'].isin(test_patients)]
```

## Results

### Baseline Model (Polynomial Regression)
- **Dataset**: 23 patients, 67 samples
- **Features**: PPG mean, std, HR
- **Performance**:
  - R² = 0.01 (test set)
  - MAE = 17.2 mg/dL
  - Conclusion: Too small, fails to generalize

### CNN-GRU Model (Target)
- **Dataset**: VitalDB (6,388 patients) or MUST (67 patients)
- **Architecture**: CNN-GRU Hybrid
- **Expected Performance**: MAE < 3 mg/dL, R² > 0.95



## References

1. "Non-Invasive Glucose Level Monitoring from PPG using a Hybrid CNN-GRU Deep Learning Network" (2024)
2. VitalDB: https://vitaldb.net
3. MUST Dataset: https://data.mendeley.com/datasets/37pm7jk7jn/3
4. Clarke Error Grid: Clinical accuracy zones for glucose monitoring

