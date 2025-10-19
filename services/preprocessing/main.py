"""
Preprocessing Service
Implements exact preprocessing from dl-model branch
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from scipy import signal
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Preprocessing Service")

MODEL_SERVICE_URL = "http://model:8002"

class PreprocessRequest(BaseModel):
    signal: List[float]

class PreprocessResponse(BaseModel):
    success: bool
    num_segments: int
    segments: Optional[List[List[float]]] = None
    quality_score: float
    error: Optional[str] = None

class PPGPreprocessor:
    """
    Exact implementation from dl-model branch
    """

    def __init__(
        self,
        sampling_rate=100,
        segment_length=1.0,
        lowcut=0.5,
        highcut=8.0,
        filter_order=4,
        peak_height_threshold=20,
        peak_distance_factor=0.8,
        similarity_threshold=0.85
    ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_samples = int(sampling_rate * segment_length)
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.peak_height_threshold = peak_height_threshold
        self.peak_distance = int(peak_distance_factor * sampling_rate)
        self.similarity_threshold = similarity_threshold

    def handle_missing_values(self, ppg_signal):
        ppg = np.asarray(ppg_signal, dtype=np.float64).copy()

        valid_mask = np.isfinite(ppg)
        if not valid_mask.any():
            return None

        missing_ratio = (~valid_mask).sum() / len(ppg)
        if missing_ratio > 0.5:
            return None

        if not valid_mask.all():
            indices = np.arange(len(ppg))
            ppg[~valid_mask] = np.interp(
                indices[~valid_mask],
                indices[valid_mask],
                ppg[valid_mask]
            )

        return ppg

    def bandpass_filter(self, ppg_signal):
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = signal.butter(self.filter_order, [low, high], btype='band')

        try:
            filtered_signal = signal.filtfilt(b, a, ppg_signal)
        except ValueError:
            return None

        return filtered_signal

    def detect_peaks(self, ppg_signal):
        peaks, _ = signal.find_peaks(
            ppg_signal,
            height=self.peak_height_threshold,
            distance=self.peak_distance
        )
        return peaks

    def extract_peak_centered_windows(self, ppg_signal, peaks):
        windows = []
        window_half = self.segment_samples // 2

        for peak in peaks:
            window_start = max(0, peak - window_half)
            window_end = min(len(ppg_signal), peak + window_half)

            if window_end - window_start == self.segment_samples:
                window = ppg_signal[window_start:window_end]
                if np.isfinite(window).all():
                    windows.append(window)

        if len(windows) == 0:
            return np.array([])

        return np.array(windows)

    def compute_template(self, windows):
        return np.mean(windows, axis=0)

    def compute_cosine_similarity(self, window, template):
        dot_product = np.sum(window * template)
        magnitude_window = np.sqrt(np.sum(window ** 2))
        magnitude_template = np.sqrt(np.sum(template ** 2))

        if magnitude_window == 0 or magnitude_template == 0:
            return 0.0

        similarity = dot_product / (magnitude_window * magnitude_template)
        return similarity

    def filter_windows_by_similarity(self, windows, template):
        filtered_windows = []

        for window in windows:
            similarity = self.compute_cosine_similarity(window, template)
            if similarity >= self.similarity_threshold:
                filtered_windows.append(window)

        if len(filtered_windows) == 0:
            return np.array([])

        return np.array(filtered_windows)

    def preprocess(self, ppg_signal, apply_template_matching=True):
        # Step 1: Handle missing values
        cleaned_signal = self.handle_missing_values(ppg_signal)
        if cleaned_signal is None:
            return None

        # Step 2: Bandpass filtering
        filtered_signal = self.bandpass_filter(cleaned_signal)
        if filtered_signal is None:
            return None

        # Step 3: Detect peaks
        peaks = self.detect_peaks(filtered_signal)
        if len(peaks) == 0:
            return None

        # Step 4: Extract windows
        windows = self.extract_peak_centered_windows(filtered_signal, peaks)
        if len(windows) == 0:
            return None

        # Step 5: Template matching
        if apply_template_matching and len(windows) > 1:
            template = self.compute_template(windows)
            filtered_windows = self.filter_windows_by_similarity(windows, template)

            if len(filtered_windows) == 0:
                return windows

            return filtered_windows

        return windows

@app.get("/")
async def root():
    return {"service": "Preprocessing Service", "status": "ready"}

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_signal(request: PreprocessRequest):
    """Preprocess PPG signal and send to model service"""
    try:
        logger.info(f"Received signal with {len(request.signal)} samples")

        # Convert to numpy
        ppg_signal = np.array(request.signal, dtype=np.float64)

        # Initialize preprocessor
        preprocessor = PPGPreprocessor(
            sampling_rate=100,
            peak_height_threshold=20,
            similarity_threshold=0.85
        )

        # Preprocess
        segments = preprocessor.preprocess(ppg_signal, apply_template_matching=True)

        if segments is None or len(segments) == 0:
            logger.warning("No valid segments extracted")
            return PreprocessResponse(
                success=False,
                num_segments=0,
                quality_score=0.0,
                error="No valid segments extracted"
            )

        logger.info(f"Extracted {len(segments)} segments")

        # Calculate quality score (based on template matching)
        quality_score = min(1.0, len(segments) / 30.0)  # Expect ~30 segments

        # Convert segments to list for JSON
        segments_list = segments.tolist()

        # Send to model service
        async with httpx.AsyncClient(timeout=60.0) as client:
            model_response = await client.post(
                f"{MODEL_SERVICE_URL}/predict",
                json={
                    "segments": segments_list,
                    "quality_score": quality_score
                }
            )

            if model_response.status_code != 200:
                logger.error(f"Model service error: {model_response.text}")

        return PreprocessResponse(
            success=True,
            num_segments=len(segments),
            segments=segments_list,
            quality_score=quality_score
        )

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
