"""
Preprocessing Service - CORRECTED VERSION
(Removed incorrect UI update code)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from scipy import signal
import logging
from typing import List, Optional, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Preprocessing Service")

# Allow overriding model URL when running locally vs docker-compose
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8002")

class PreprocessRequest(BaseModel):
    signal: List[float]

class PreprocessResponse(BaseModel):
    success: bool
    num_segments: int
    segments: Optional[List[List[float]]] = None
    quality_score: float
    error: Optional[str] = None

# --- Respiratory pipeline schemas ---
class RespiratoryPreprocessRequest(BaseModel):
    # Raw PPG samples
    signal: List[float]
    # Optional sampling rate override (Hz); defaults to preprocessor.sampling_rate
    sampling_rate: Optional[float] = None

class RespiratoryPreprocessResponse(BaseModel):
    success: bool
    resp_rate_bpm: Optional[float] = None
    resp_freq_hz: Optional[float] = None
    esqi: Optional[float] = None
    entropy: Optional[float] = None
    peaks_count: int = 0
    method_used: Optional[str] = None
    # Limited-length arrays to avoid huge responses
    freqs: Optional[List[float]] = None
    psd: Optional[List[float]] = None
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
        similarity_threshold=0.70,
        # Respiratory-focused extras
        resp_lowcut=0.1,
        resp_highcut=0.4,
        rb: float = 1.0,
        l_lt: float = 0.0,
        hampel_window_size: int = 15,
        hampel_threshold: float = 3.0,
        ibi_resample_hz: float = 4.0
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
        # Respiratory extras
        self.resp_lowcut = resp_lowcut
        self.resp_highcut = resp_highcut
        self.rb = rb
        self.l_lt = l_lt
        self.hampel_window_size = int(hampel_window_size if hampel_window_size % 2 == 1 else hampel_window_size + 1)
        self.hampel_threshold = hampel_threshold
        self.ibi_resample_hz = ibi_resample_hz

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

    def bandpass_filter_resp(self, ppg_signal):
        """Bandpass for respiratory component (defaults 0.1–0.4 Hz)."""
        nyquist = 0.5 * self.sampling_rate
        low = self.resp_lowcut / nyquist
        high = self.resp_highcut / nyquist

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

    def detect_peaks_dynamic(self, ppg_signal):
        """
        Heart-beat oriented peak detector with dynamic thresholds.
        Uses prominence to be robust across amplitudes.
        """
        distance = int(0.3 * self.sampling_rate)  # ~200 bpm max
        prominence = max(0.1 * np.nanstd(ppg_signal), 1e-6)
        try:
            peaks, _ = signal.find_peaks(ppg_signal, distance=distance, prominence=prominence)
        except Exception:
            peaks = np.array([], dtype=int)
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

    # --------- Respiratory helpers ---------
    def peak_enhancement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        rng = max(x_max - x_min, 1e-12)
        norm = (x - x_min) / rng
        return self.rb * norm + self.l_lt

    def hampel_filter(self, x: np.ndarray) -> np.ndarray:
        """Sample-based Hampel filter; replaces outliers with local median.
        Window size is in samples and will be forced odd.
        """
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        k = self.hampel_window_size // 2
        y = x.copy()
        if n == 0:
            return y
        # Precompute rolling medians using simple sliding window (O(n*k)) for robustness
        for i in range(k, n - k):
            w = x[i - k:i + k + 1]
            med = np.median(w)
            mad = np.median(np.abs(w - med))
            sigma = 1.4826 * mad + 1e-12
            if np.abs(x[i] - med) > self.hampel_threshold * sigma:
                y[i] = med
        return y

    def compute_esqi(self, x: np.ndarray) -> Tuple[float, float]:
        """Entropy-based signal quality index.
        Returns (entropy, esqi) where esqi = 1 - normalized_entropy in [0,1].
        """
        x = np.asarray(x, dtype=np.float64)
        power = x ** 2
        total = np.sum(power) + 1e-12
        p = power / total
        entropy = -np.sum(p * np.log(p + 1e-12))
        max_entropy = np.log(len(x) + 1e-12)
        entropy_norm = min(entropy / (max_entropy + 1e-12), 1.0)
        esqi = 1.0 - entropy_norm
        return float(entropy), float(esqi)

    def compute_ibis(self, peaks: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inter-beat intervals (seconds) and their sample times (seconds)."""
        if peaks is None or len(peaks) < 2:
            return np.array([]), np.array([])
        times = peaks / float(fs)
        ibis = np.diff(times)  # seconds between beats
        # assign IBI time at mid-point between two peaks
        ibi_times = times[:-1] + ibis / 2.0
        return ibi_times, ibis

    def resample_series(self, t: np.ndarray, y: np.ndarray, target_fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Resample uneven series y(t) to even grid at target_fs using linear interp."""
        if len(t) == 0 or len(y) == 0:
            return np.array([]), np.array([]), target_fs
        t0, t1 = float(t[0]), float(t[-1])
        if t1 <= t0:
            return np.array([]), np.array([]), target_fs
        n_samples = int(np.floor((t1 - t0) * target_fs)) + 1
        t_even = np.linspace(t0, t1, n_samples)
        y_even = np.interp(t_even, t, y)
        return t_even, y_even, target_fs

    def welch_psd(self, x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Welch PSD per supplied paper parameters.
        - Window: Hann
        - Overlap: 50%
        - nfft: length of data
        - Scaling: density
        - Averaging: mean
        """
        x = np.asarray(x, dtype=np.float64)
        if len(x) < 8:
            return np.array([]), np.array([])
        n = len(x)
        nperseg = n  # length of data
        noverlap = nperseg // 2
        freqs, psd = signal.welch(
            x,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=n,
            scaling='density',
            average='mean'
        )
        return freqs, psd

    def resp_rate_from_psd(self, freqs: np.ndarray, psd: np.ndarray, band=(0.1, 0.4)) -> Tuple[Optional[float], Optional[float]]:
        if freqs.size == 0 or psd.size == 0:
            return None, None
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return None, None
        idx = np.argmax(psd[mask])
        f_band = freqs[mask]
        f_max = float(f_band[idx])
        return f_max * 60.0, f_max

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
            sampling_rate=100,              # 100 Hz
            segment_length=1.0,             # 1-second windows
            lowcut=0.5,                     # ← FIXED: Was 0.1, should be 0.5 Hz
            highcut=8.0,                    # Remove frequencies > 8 Hz
            filter_order=4,                 # Butterworth filter order
            peak_height_threshold=20,       # Minimum peak height (adjust if needed)
            peak_distance_factor=0.8,       # Min distance between peaks
            similarity_threshold=0.70       # Template matching threshold (70%)
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

        # Calculate quality score (based on number of segments)
        quality_score = min(1.0, len(segments) / 30.0)  # Expect ~30 segments

        # Convert segments to list for JSON
        segments_list = segments.tolist()

        # Send to model service (if you have one)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                model_response = await client.post(
                    f"{MODEL_SERVICE_URL}/predict",
                    json={
                        "segments": segments_list,
                        "quality_score": quality_score
                    }
                )

                if model_response.status_code == 200:
                    logger.info("✓ Model service processed segments")
                else:
                    logger.warning(f"Model service returned {model_response.status_code}")
                    
        except Exception as model_error:
            logger.warning(f"Model service not available: {model_error}")
            # Continue anyway - preprocessing succeeded

        # Return preprocessing result
        return PreprocessResponse(
            success=True,
            num_segments=len(segments),
            segments=segments_list,
            quality_score=quality_score
        )

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess_respiratory", response_model=RespiratoryPreprocessResponse)
async def preprocess_respiratory(request: RespiratoryPreprocessRequest):
    """
    Respiratory-focused pipeline that:
    1) Handles missing values
    2) Bandpass (0.1–0.4 Hz)
    3) Peak enhancement
    4) Hampel outlier removal
    5) ESQI entropy metric
    6) Heart-beat peak detection on HEART BAND (0.5–8 Hz) to form IBIs
    7) Resample IBI series to even grid and compute Welch PSD
    8) RespR = argmax PSD within [0.1, 0.4] Hz, converted to bpm

    Falls back to PSD of respiratory-band filtered signal if IBI series is insufficient.
    """
    try:
        raw = np.array(request.signal, dtype=np.float64)
        logger.info(f"[Resp] received {raw.size} samples")

        # Use provided sampling rate if any
        pre = PPGPreprocessor()
        if request.sampling_rate is not None:
            pre.sampling_rate = float(request.sampling_rate)
            pre.segment_samples = int(pre.sampling_rate * pre.segment_length)

        # 1) Missing values
        cleaned = pre.handle_missing_values(raw)
        if cleaned is None:
            return RespiratoryPreprocessResponse(success=False, peaks_count=0, error="Signal invalid after missing-value handling")

        # 2) Respiratory bandpass
        resp_filt = pre.bandpass_filter_resp(cleaned)
        if resp_filt is None:
            return RespiratoryPreprocessResponse(success=False, peaks_count=0, error="Respiratory bandpass failed")

        # 3) Peak enhancement
        enhanced = pre.peak_enhancement(resp_filt)

        # 4) Hampel filter
        denoised = pre.hampel_filter(enhanced)

        # 5) ESQI from the denoised respiratory component
        entropy, esqi = pre.compute_esqi(denoised)

        # 6) Heart-beat oriented peak detection on HEART BAND to get IBIs
        pre_heart = PPGPreprocessor(sampling_rate=pre.sampling_rate)
        heart_band = pre_heart.bandpass_filter(cleaned)
        peaks = pre.detect_peaks_dynamic(heart_band) if heart_band is not None else np.array([])

        # 7) Form IBI series and resample
        ibi_t, ibi = pre.compute_ibis(peaks, pre.sampling_rate)
        method_used = "ibi-welch"
        freqs = np.array([])
        psd = np.array([])
        resp_rate_bpm: Optional[float] = None
        resp_freq_hz: Optional[float] = None

        if len(ibi) >= 8:
            t_even, ibi_even, fs_even = pre.resample_series(ibi_t, ibi, pre.ibi_resample_hz)
            freqs, psd = pre.welch_psd(ibi_even - np.mean(ibi_even), fs_even)
            if freqs.size and psd.size:
                resp_rate_bpm, resp_freq_hz = pre.resp_rate_from_psd(freqs, psd, band=(pre.resp_lowcut, pre.resp_highcut))

        # 7b) Fallback: use the respiratory-band signal directly
        if resp_rate_bpm is None:
            method_used = "resp-signal-welch"
            freqs, psd = pre.welch_psd(denoised - np.mean(denoised), pre.sampling_rate)
            if freqs.size and psd.size:
                rr_bpm, rr_hz = pre.resp_rate_from_psd(freqs, psd, band=(pre.resp_lowcut, pre.resp_highcut))
                resp_rate_bpm, resp_freq_hz = rr_bpm, rr_hz

        if resp_rate_bpm is None:
            return RespiratoryPreprocessResponse(
                success=False,
                peaks_count=int(len(peaks)),
                esqi=esqi,
                entropy=entropy,
                error="Unable to estimate respiratory rate"
            )

        # Trim PSD vectors if very long to keep response size sane
        MAX_PSD_POINTS = 2048
        if freqs.size > MAX_PSD_POINTS:
            idx = np.linspace(0, freqs.size - 1, MAX_PSD_POINTS).astype(int)
            freqs_out = freqs[idx].tolist()
            psd_out = psd[idx].tolist()
        else:
            freqs_out = freqs.tolist()
            psd_out = psd.tolist()

        return RespiratoryPreprocessResponse(
            success=True,
            resp_rate_bpm=float(resp_rate_bpm),
            resp_freq_hz=float(resp_freq_hz) if resp_freq_hz is not None else None,
            esqi=float(esqi),
            entropy=float(entropy),
            peaks_count=int(len(peaks)),
            method_used=method_used,
            freqs=freqs_out,
            psd=psd_out,
        )
    except Exception as e:
        logger.error(f"Respiratory preprocessing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)