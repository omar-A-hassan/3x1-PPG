"""
Model Service
Loads best_model.pt and runs inference on preprocessed PPG segments
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import httpx
import numpy as np
import torch
import sys
import logging
from pathlib import Path
from typing import List, Optional
import os
import pandas as pd

# Set PyTorch to avoid potential issues in Docker
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Service")
app = FastAPI(title="Model Service")

# UI callback endpoint (env override for local runs)
UI_SERVICE_URL = os.getenv("UI_SERVICE_URL", "http://localhost:8003")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'ts2vec'))

class PredictRequest(BaseModel):
    segments: List[List[float]]
    quality_score: float

class PredictResponse(BaseModel):
    success: bool
    glucose_prediction: Optional[float] = None
    num_segments: int
    quality_score: float
    device: str
    error: Optional[str] = None

class ModelInference:
    """Model inference handler with MPS/CPU fallback"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self._load_model()

    def _get_device(self):
        """Get best available device: MPS > CPU"""
        if torch.backends.mps.is_available():
            logger.info("MPS backend available, using MPS")
            return torch.device("mps")
        else:
            logger.info("MPS not available, using CPU")
            return torch.device("cpu")

    def _load_model(self):
        """Load model from checkpoint"""
        try:
            # Import model class
            from src.models.ts2vec_ppg import build_tsencoder_ppg

            logger.info(f"Loading model from {self.model_path}")

            # Build model architecture
            self.model = build_tsencoder_ppg(
                input_dims=1,
                output_dims=320,
                hidden_dims=64,
                depth=10,
                dropout=0.1
            )

            # Load weights
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            # Count parameters
            params = self.model.count_parameters()
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Total parameters: {params['total']:,}")
            logger.info(f"Encoder parameters: {params['encoder']:,}")
            logger.info(f"Regression head parameters: {params['regression_head']:,}")

            # Skip warmup - causes segfault with PyTorch 2.1.0 in Docker
            logger.info("Model loaded and ready for inference")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def predict(self, segments: np.ndarray) -> float:
        """
        Run inference on preprocessed segments

        Args:
            segments: numpy array of shape (num_segments, segment_length)

        Returns:
            glucose prediction in mg/dL
        """
        try:
            logger.info(f"[PREDICT] Starting prediction for {len(segments)} segments")
            logger.info(f"[PREDICT] Segments shape: {segments.shape}, dtype: {segments.dtype}")

            # Convert to tensor
            logger.info("[PREDICT] Converting to tensor...")
            x = torch.tensor(segments, dtype=torch.float32)
            logger.info(f"[PREDICT] Converted to tensor, shape: {x.shape}")

            # Add batch dimension if needed
            if x.dim() == 2:
                # (num_segments, segment_length) -> (num_segments, segment_length, 1)
                logger.info("[PREDICT] Adding feature dimension...")
                x = x.unsqueeze(-1)
                logger.info(f"[PREDICT] Added feature dimension, new shape: {x.shape}")

            # Move to device
            logger.info(f"[PREDICT] Moving to device: {self.device}")
            x = x.to(self.device)
            logger.info(f"[PREDICT] Moved to device successfully")

            # Run inference - use no_grad instead of inference_mode (known bug in 2.1.0)
            logger.info("[PREDICT] About to run model.forward()...")
            try:
                with torch.no_grad():
                    logger.info("[PREDICT] Inside torch.no_grad context")
                    # Pass mask='all_true' to avoid mask generation issues
                    logger.info("[PREDICT] Calling model(x, mask='all_true')...")
                    predictions = self.model(x, mask='all_true')  # (num_segments, 1)
                    logger.info(f"[PREDICT] Model returned, predictions shape: {predictions.shape}")
            except Exception as model_error:
                logger.error(f"[PREDICT] Model forward pass failed: {model_error}", exc_info=True)
                logger.error(f"[PREDICT] Input shape was: {x.shape}, device: {x.device}")
                raise

            # Average predictions across segments
            logger.info("[PREDICT] Averaging predictions...")
            glucose_pred = predictions.mean().item()
            logger.info(f"[PREDICT] Got final value: {glucose_pred:.1f} mg/dL")

            logger.info(f"[PREDICT] SUCCESS: Predicted glucose: {glucose_pred:.1f} mg/dL from {len(segments)} segments")

            return glucose_pred

        except Exception as e:
            logger.error(f"[PREDICT] Inference failed: {e}", exc_info=True)
            raise

# Global model instance
model_inference = None


@app.on_event("startup")
async def startup_event():
    global model_inference
    model_path = Path(r"C:\Users\nazeh\BioInfo Trials\3x1-PPG\Dev\3x1-PPG\services\model\best_model.pt")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Model file found at: {model_path}")

    model_inference = ModelInference(str(model_path))
    logger.info("✅ Model loaded successfully and ready for inference.")

    try:
        # ✅ Import the same model builder used in Docker
        from src.models.ts2vec_ppg import build_tsencoder_ppg

        # Build architecture (must match what was trained)
        model = build_tsencoder_ppg(
            input_dims=1,
            output_dims=320,
            hidden_dims=64,
            depth=10,
            dropout=0.1
        )

        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()  # ready for inference

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
async def root():
    return {
        "service": "Model Service",
        "status": "ready",
        "device": str(next(model_inference.parameters()).device) if model_inference else "not initialized"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "device": str(model_inference.device),
        "model_path": model_inference.model_path
    }


@app.post("/predict_from_csv", response_model=PredictResponse)
async def predict_from_csv(file: UploadFile = File(...), quality_score: float = 1.0):
    """
    Run inference on uploaded CSV file containing preprocessed PPG segments.
    Each row is a segment.
    """
    try:
        if model_inference is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Load CSV into DataFrame
        df = pd.read_csv(file.file)
        segments = df.values.astype(np.float32)
        
        logger.info(f"Loaded CSV with shape {segments.shape}")

        if segments.ndim == 1:
            segments = np.expand_dims(segments, axis=0)

        # Run inference
        glucose_pred = model_inference.predict(segments)

        return PredictResponse(
            success=True,
            glucose_prediction=glucose_pred,
            num_segments=len(segments),
            quality_score=quality_score,
            device=str(model_inference.device)
        )
        
    except Exception as e:
        logger.error(f"Prediction from CSV failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Run inference on preprocessed segments"""
    try:
        if model_inference is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        logger.info(f"Received {len(request.segments)} segments for inference")
        
        # Convert to numpy
        segments = np.array(request.segments, dtype=np.float32)
        
        if len(segments) == 0:
            return PredictResponse(
                success=False,
                num_segments=0,
                quality_score=request.quality_score,
                device=str(model_inference.device),
                error="No segments provided"
            )
        
        # Run inference
        logger.info("About to call model_inference.predict...")
        try:
            glucose_pred = model_inference.predict(segments)
            logger.info(f"Prediction successful: {glucose_pred:.2f} mg/dL")
        except Exception as pred_error:
            logger.error(f"Prediction failed: {pred_error}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(pred_error)}")

        # Send to UI service
        logger.info("Sending result to UI service...")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                ui_response = await client.post(
                    f"{UI_SERVICE_URL}/update_result",
                    json={
                        "glucose": glucose_pred,
                        "num_segments": len(segments),
                        "quality_score": request.quality_score,
                        "device": str(model_inference.device)
                    }
                )
                
                if ui_response.status_code != 200:
                    logger.warning(f"UI service update failed: {ui_response.text}")
                else:
                    logger.info("Successfully sent result to UI")
                    
        except Exception as e:
            logger.warning(f"Failed to send to UI service: {e}")
        
        return PredictResponse(
            success=True,
            glucose_prediction=glucose_pred,
            num_segments=len(segments),
            quality_score=request.quality_score,
            device=str(model_inference.device)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run with workers=1 to avoid multiprocessing issues with PyTorch in Docker
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)
