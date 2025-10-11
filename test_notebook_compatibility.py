#!/usr/bin/env python3
"""
Test script to validate notebook compatibility with Trainer API.

This script verifies that the Trainer class supports the API expected by
the ts2vec-training.ipynb notebook.

Usage:
    python test_notebook_compatibility.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from src.models import build_tsencoder_ppg, TS2VEC_AVAILABLE
        from src.preprocessing import PPGPreprocessor
        from src.training import Trainer
        from src.evaluation import compute_all_metrics, print_metrics
        
        if not TS2VEC_AVAILABLE:
            print("❌ FAIL: TS2Vec not available")
            print("   Please ensure ts2vec submodule is initialized:")
            print("   git submodule update --init --recursive")
            return False
        
        print("✅ PASS: All imports successful")
        return True
    except ImportError as e:
        print(f"❌ FAIL: Import error: {e}")
        return False

def test_trainer_api():
    """Test that Trainer supports the notebook API."""
    print("\nTesting Trainer API...")
    try:
        from src.models import build_tsencoder_ppg
        from src.training import Trainer
        
        # Test 1: checkpoint_dir parameter
        model = build_tsencoder_ppg(input_dims=1, output_dims=64, hidden_dims=32, depth=3)
        trainer = Trainer(model=model, device='cpu', checkpoint_dir='/tmp/test')
        
        if not hasattr(trainer, 'checkpoint_dir'):
            print("❌ FAIL: Trainer missing checkpoint_dir attribute")
            return False
        
        # Test 2: best_val_mae attribute
        if not hasattr(trainer, 'best_val_mae'):
            print("❌ FAIL: Trainer missing best_val_mae attribute")
            return False
        
        # Test 3: train() method exists
        if not hasattr(trainer, 'train') or not callable(getattr(trainer, 'train')):
            print("❌ FAIL: Trainer missing train() method")
            return False
        
        print("✅ PASS: Trainer API compatible with notebook")
        return True
    except Exception as e:
        print(f"❌ FAIL: Trainer API test error: {e}")
        return False

def test_training_workflow():
    """Test complete training workflow as in notebook."""
    print("\nTesting training workflow...")
    try:
        from src.models import build_tsencoder_ppg
        from src.training import Trainer
        from src.evaluation import compute_all_metrics
        
        # Create small test dataset
        X_train = torch.randn(50, 100)
        y_train = torch.randn(50, 1) * 10 + 100
        X_val = torch.randn(10, 100)
        y_val = torch.randn(10, 1) * 10 + 100
        
        # Build and train model
        device = 'cpu'
        model = build_tsencoder_ppg(input_dims=1, output_dims=32, hidden_dims=16, depth=3)
        
        trainer = Trainer(
            model=model,
            device=device,
            checkpoint_dir='/tmp/workflow_test'
        )
        
        # Train for 1 epoch (quick test)
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=16,
            epochs=1,
            learning_rate=0.001,
            patience=10,
            min_delta=0.5
        )
        
        # Load checkpoint
        trainer.load_checkpoint('/tmp/workflow_test/best_model.pt')
        
        # Test inference
        X_test = torch.randn(10, 100)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test.to(device)).cpu().numpy().squeeze()
        
        if y_pred.shape[0] != 10:
            print(f"❌ FAIL: Unexpected prediction shape: {y_pred.shape}")
            return False
        
        print("✅ PASS: Complete training workflow successful")
        return True
    except Exception as e:
        print(f"❌ FAIL: Training workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Notebook Compatibility Test Suite")
    print("="*60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Trainer API", test_trainer_api()))
    results.append(("Training Workflow", test_training_workflow()))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Notebook should work correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
