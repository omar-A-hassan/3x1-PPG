"""
Training Module for CNN-GRU Glucose Prediction Model

Implements training loop with:
- MSE/MAE loss functions
- Adam optimizer with learning rate scheduling
- Early stopping
- Model checkpointing

Based on: "Non-Invasive Glucose Level Monitoring from PPG using a
Hybrid CNN-GRU Deep Learning Network" (2024)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


class Trainer:
    """
    Trainer class for CNN-GRU model.

    Args:
        model (nn.Module): CNN-GRU model
        device (str): Device to train on ('cuda' or 'cpu')
        learning_rate (float): Initial learning rate (default: 0.001)
        weight_decay (float): L2 regularization (default: 1e-5)
    """

    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def compute_mae(self, predictions, targets):
        """Compute Mean Absolute Error"""
        return torch.mean(torch.abs(predictions - targets)).item()

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            epoch_mae += self.compute_mae(predictions, batch_y)
            n_batches += 1

        return epoch_loss / n_batches, epoch_mae / n_batches

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                # Metrics
                epoch_loss += loss.item()
                epoch_mae += self.compute_mae(predictions, batch_y)
                n_batches += 1

        return epoch_loss / n_batches, epoch_mae / n_batches

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=100,
        early_stopping_patience=15,
        save_dir='checkpoints',
        verbose=True
    ):
        """
        Train model with early stopping and checkpointing.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Maximum number of epochs
            early_stopping_patience (int): Stop if no improvement for N epochs
            save_dir (str): Directory to save checkpoints
            verbose (bool): Print training progress
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_mae = self.train_epoch(train_loader)

            # Validate
            val_loss, val_mae = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(current_lr)

            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f} mg/dL")
                print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f} mg/dL")
                print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0

                # Save checkpoint
                checkpoint_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'history': self.history
                }, checkpoint_path)

                if verbose:
                    print(f"  → Best model saved (Val MAE: {val_mae:.2f} mg/dL)")

            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best model from epoch {self.best_epoch+1} with Val MAE: {self.history['val_mae'][self.best_epoch]:.2f} mg/dL")
                break

        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"Val MAE: {checkpoint['val_mae']:.2f} mg/dL")


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=32,
    epochs=100,
    learning_rate=0.001,
    device='cuda',
    save_dir='checkpoints'
):
    """
    Convenience function to train CNN-GRU model.

    Args:
        model: CNN-GRU model instance
        X_train (np.ndarray): Training PPG segments (n_samples, seq_length)
        y_train (np.ndarray): Training glucose values (n_samples,)
        X_val (np.ndarray): Validation PPG segments
        y_val (np.ndarray): Validation glucose values
        batch_size (int): Batch size
        epochs (int): Maximum epochs
        learning_rate (float): Learning rate
        device (str): 'cuda' or 'cpu'
        save_dir (str): Directory to save checkpoints

    Returns:
        Trainer instance with training history
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize trainer
    trainer = Trainer(model, device=device, learning_rate=learning_rate)

    # Train
    print("="*60)
    print("Training CNN-GRU Model")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*60)

    trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        save_dir=save_dir
    )

    return trainer


if __name__ == "__main__":
    from ..models import build_cnn_gru

    print("="*60)
    print("Testing CNN-GRU Trainer")
    print("="*60)

    # Generate synthetic data
    n_train = 1000
    n_val = 200
    seq_length = 100

    X_train = np.random.randn(n_train, seq_length).astype(np.float32)
    y_train = np.random.uniform(70, 180, n_train).astype(np.float32)
    X_val = np.random.randn(n_val, seq_length).astype(np.float32)
    y_val = np.random.uniform(70, 180, n_val).astype(np.float32)

    # Build model
    model = build_cnn_gru(input_length=seq_length)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Train for 3 epochs (quick test)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    trainer = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=32,
        epochs=3,
        learning_rate=0.001,
        device=device,
        save_dir='test_checkpoints'
    )

    print("\n" + "="*60)
    print("Trainer working correctly")
    print("="*60)
