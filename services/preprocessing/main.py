"""Preprocessing Service"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import numpy as np
from scipy import signal
import logging
from typing import List, Optional, Tuple
import os
from functools import wraps
from scipy.signal import hilbert, welch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Preprocessing Service")

# Allow overriding model URL when running locally vs docker-compose
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8002")


# ============================================================================
# ERROR HANDLING DECORATOR
# ============================================================================

def handle_preprocessing_errors(operation_name: str):
    """Decorator to standardize error handling across preprocessing endpoints.
    
    Args:
        operation_name: Name of the operation for logging (e.g., "Preprocessing", "Respiratory preprocessing")
        
    Returns:
        Decorator that wraps endpoint functions with consistent error handling
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        return wrapper
    return decorator

class PreprocessRequest(BaseModel):
    signal: List[float]

class PreprocessResponse(BaseModel):
    success: bool
    num_segments: int
    segments: Optional[List[List[float]]] = None
    quality_score: float
    glucose: Optional[float] = None
    model_device: Optional[str] = None
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
    # Diagnostics
    alt_peak_bpm: Optional[float] = None
    peak_prominence: Optional[float] = None
    dominance_ratio: Optional[float] = None
    used_fallback: Optional[bool] = None
    # Limited-length arrays to avoid huge responses
    freqs: Optional[List[float]] = None
    psd: Optional[List[float]] = None
    error: Optional[str] = None

# ---------------------------------------------------------------------------
            # PPG Preprocessor initialization / All parameters
# ---------------------------------------------------------------------------

class PPGPreprocessor:
    """
    Exact implementation from dl-model branch
    """

    def __init__(
        self,
        sampling_rate=50,
        segment_length=1.0,
        lowcut=0.4,  # Relaxed from 0.5 to 0.4 Hz for better 50-100 Hz compatibility
        highcut=8.0,
        filter_order=4,
        similarity_threshold=0.70,
        # Respiratory-focused extras
        resp_lowcut=0.15,
        resp_highcut=0.5,
        rb: float = 1.0,
        l_lt: float = 0.0,
        hampel_window_size: int = 220,
        hampel_threshold: float = 3.0,
        ibi_resample_hz: float = 20.0  # Increased from 10.0 for better frequency resolution
    ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_samples = int(sampling_rate * segment_length)
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
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

    def bandpass_filter(
        self,
        ppg_signal,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        mode: str = "heart",
    ):
        """Generic bandpass filter for PPG.
        Args:
            ppg_signal: array-like PPG samples.
            lowcut: optional low cutoff in Hz; if None uses defaults per mode.
            highcut: optional high cutoff in Hz; if None uses defaults per mode.
            mode: 'heart' to use (self.lowcut, self.highcut) or 'resp' to use
                  (self.resp_lowcut, self.resp_highcut). Custom low/high override mode.
        Returns:
            np.ndarray filtered signal, or None on failure.
        """
        # Resolve band
        if lowcut is None or highcut is None:
            if mode == "resp":
                lc = self.resp_lowcut if lowcut is None else lowcut
                hc = self.resp_highcut if highcut is None else highcut
            else:
                lc = self.lowcut if lowcut is None else lowcut
                hc = self.highcut if highcut is None else highcut
        else:
            lc, hc = lowcut, highcut

        nyquist = 0.5 * self.sampling_rate
        low = float(lc) / nyquist
        high = float(hc) / nyquist

        # Guard against invalid normalized frequencies
        if not (0.0 < low < high < 1.0):
            return None

        b, a = signal.butter(self.filter_order, [low, high], btype='band')
        try:
            filtered_signal = signal.filtfilt(b, a, ppg_signal)
        except ValueError:
            return None
        return filtered_signal

    def detect_peaks_dynamic(self, ppg_signal):
        """
        Heart-beat oriented peak detector with dynamic thresholds.
        Uses prominence to be robust across amplitudes.
        """
        distance = int(0.4 * self.sampling_rate)  # ~200 bpm max
        prominence = max(0.05 * np.nanstd(ppg_signal), 1e-6)  # Reduced from 0.1 for better sensitivity
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

    # --------- Omar PPG pipeline ---------

    def preprocess(self, ppg_signal, apply_template_matching=True):
        # Step 1: Handle missing values
        cleaned_signal = self.handle_missing_values(ppg_signal)
        if cleaned_signal is None:
            return None

        # Step 2: Bandpass filtering
        filtered_signal = self.bandpass_filter(cleaned_signal)
        if filtered_signal is None:
            return None

        # Step 3: Detect peaks (dynamic, robustness across amplitudes)
        peaks = self.detect_peaks_dynamic(filtered_signal)
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
        enchancement = self.rb * norm + self.l_lt
        return pow(enchancement,2)

    def hampel_filter(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Hampel filter; replaces outliers with local median.
        Uses scipy.ndimage.median_filter for efficient computation.
        """
        from scipy.ndimage import median_filter
        
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        if n == 0:
            return x
        
        k = self.hampel_window_size
        # Compute rolling median using fast convolution-based filter
        med = median_filter(x, size=k, mode='reflect')
        # Compute median absolute deviation
        mad = median_filter(np.abs(x - med), size=k, mode='reflect')
        sigma = 1.4826 * mad + 1e-12
        
        # Replace outliers with median values
        mask = np.abs(x - med) > self.hampel_threshold * sigma
        y = x.copy()
        y[mask] = med[mask]
        return y

    def compute_esqi(self, x: np.ndarray) -> Tuple[float, float]:
        """Entropy-based signal quality index matching the paper formula."""
        x = np.asarray(x, dtype=np.float64)
        x_squared = x ** 2
    
        # Compute entropy directly on squared values (no normalization)
        entropy = -np.sum(x_squared * np.log(x_squared + 1e-12))
        esqi = 1.0 - entropy
        return float(entropy), float(esqi)

    def compute_ibis(self, peaks: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inter-beat intervals (seconds) and their sample times (seconds)."""
        if peaks is None or len(peaks) < 2:
            return np.array([]), np.array([])
        times = peaks / float(fs)
        ibis = np.diff(times)  # seconds between beats
        # assign IBI time at mid-point between two peaks
        ibi_times = times[:-1] + ibis / 2.0
        logger.info("n_IBI: %d", len(ibis))
        logger.info("IBI mean (ms): %.1f std (ms): %.1f", np.mean(ibis)*1000, np.std(ibis)*1000)
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
        """Welch PSD with optimized segmentation for variance reduction.
        - Window: Hann
        - Overlap: 50%
        - Segment length: 30 seconds (or full length if shorter)
        - Scaling: density
        - Averaging: mean
        """
        x = np.asarray(x, dtype=np.float64)
        if len(x) < 8:
            return np.array([]), np.array([])
        n = len(x)
        nperseg = min(n, int(30 * fs))  # 30-second segments for better frequency resolution
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

    def select_resp_peak(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        band: Tuple[float, float] = (0.1, 0.4),
        min_bpm: float = 6.0,
        max_bpm: float = 30.0,
        min_prom_rel: float = 0.12,
        dominance_ratio: float = 1.2,
        min_width_bins: int = 1,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Robustly select respiratory peak using prominence and dominance checks.

        Returns (bpm, hz, peak_prominence, dominance_ratio) or (None, ...)
        """
        if freqs.size == 0 or psd.size == 0:
            return None, None, None, None
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return None, None, None, None
        f = freqs[mask]
        p = psd[mask]
        if p.size < 3:
            idx = int(np.argmax(p))
            f0 = float(f[idx])
            bpm = f0 * 60.0
            if bpm < min_bpm or bpm > max_bpm:
                return None, None, None, None
            return bpm, f0, None, None
        # peak finding with prominence relative to max
        rel_prom = max(min_prom_rel * (np.nanmax(p) - np.nanmin(p)), 1e-12)
        try:
            pk_idx, props = signal.find_peaks(p, prominence=rel_prom, width=min_width_bins)
        except Exception:
            pk_idx = np.array([], dtype=int)
            props = {}
        if pk_idx.size == 0:
            # fallback to argmax within band but enforce bpm limits
            i = int(np.argmax(p))
            f0 = float(f[i])
            bpm = f0 * 60.0
            if bpm < min_bpm or bpm > max_bpm:
                return None, None, None, None
            return bpm, f0, None, None
        # choose by maximum prominence
        prominences = props.get('prominences', np.ones(pk_idx.size))
        order = int(np.argmax(prominences))
        main_idx = pk_idx[order]
        prom_main = float(prominences[order]) if np.ndim(prominences) else float(prominences)
        # compute dominance ratio vs second-best if available
        dom = None
        if pk_idx.size > 1:
            sorted_prom = np.sort(prominences)[::-1]
            dom = float(sorted_prom[0] / max(sorted_prom[1], 1e-12))
            if dom < dominance_ratio:
                # not dominant enough → consider unstable; do not return
                return None, None, float(prom_main), float(dom)
        f0 = float(f[main_idx])
        bpm = f0 * 60.0
        if bpm < min_bpm or bpm > max_bpm:
            return None, None, float(prom_main), float(dom) if dom is not None else None
        return bpm, f0, float(prom_main), float(dom) if dom is not None else None

@app.get("/")
async def root():
    return {"service": "Preprocessing Service", "status": "ready"}

@app.post("/preprocess", response_model=PreprocessResponse)
@handle_preprocessing_errors("Preprocessing")
async def preprocess_signal(request: PreprocessRequest):
    """Preprocess PPG signal and send to model service"""
    logger.info(f"Received signal with {len(request.signal)} samples")

    ppg_signal = np.array(request.signal, dtype=np.float64)
    
    # Initialize preprocessor with glucose-specific parameters
    preprocessor = PPGPreprocessor(
        sampling_rate=50,      # Hz - standard PPG sampling rate
        lowcut=0.4             # Hz - heart-rate bandpass lower bound
    )
    
    # Run full preprocessing pipeline: bandpass → peak detection → segmentation
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
    
    # Calculate quality score based on segment count (30 segments ≈ quality score of 1.0)
    quality_score = min(1.0, len(segments) / 30.0)
    
    # Convert segments to list for JSON serialization
    segments_list = segments.tolist()

    glucose_prediction: Optional[float] = None
    model_device: Optional[str] = None

    # Forward segments to model service for glucose prediction
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
                try:
                    model_payload = model_response.json()
                    glucose_prediction = model_payload.get("glucose_prediction")
                    model_device = model_payload.get("device")
                    if glucose_prediction is not None:
                        logger.info(
                            f"Predicted glucose: {float(glucose_prediction):.1f} mg/dL"
                        )
                except ValueError:
                    logger.warning("Model service returned non-JSON response")
                except Exception as parse_error:
                    logger.warning(f"Failed to parse model response: {parse_error}")
            else:
                logger.warning(f"Model service returned {model_response.status_code}")
    except Exception as model_error:
        # Continue anyway - preprocessing succeeded, model prediction is optional
        logger.warning(f"Model service not available: {model_error}")

    return PreprocessResponse(
        success=True,
        num_segments=len(segments),
        segments=segments_list,
        quality_score=quality_score,
        glucose=glucose_prediction,
        model_device=model_device
    )


@app.post("/preprocess_respiratory", response_model=RespiratoryPreprocessResponse)
@handle_preprocessing_errors("Respiratory preprocessing")
async def preprocess_respiratory(request: RespiratoryPreprocessRequest):
    """
    Respiratory rate estimation pipeline:
    1) Missing values handling
    2) Respiratory bandpass filter (0.1-0.5 Hz)
    3) Peak enhancement
    4) Hampel filter for outlier removal
    6) IBI extraction via heart-rate band peak detection
    7) PSD analysis (IBI-based if available, else signal-based)
    8) Respiratory rate estimation from dominant frequency
    
    Returns: respiratory_rate (bpm), peaks_count, dominant_freq, and optional IBI/PSD arrays
    """
    raw = np.array(request.signal, dtype=np.float64)
    logger.info(f"[Resp] received {raw.size} samples")
    
    # Initialize preprocessor with default parameters
    pre = PPGPreprocessor()
    
    # Use provided sampling rate if any
    if request.sampling_rate is not None:
        pre.sampling_rate = int(request.sampling_rate)
        pre.segment_samples = int(pre.sampling_rate * pre.segment_length)

    # 1) Handle missing values (interpolation/removal)
    cleaned = pre.handle_missing_values(raw)
    if cleaned is None:
        return RespiratoryPreprocessResponse(success=False, peaks_count=0, error="Signal invalid after missing-value handling")
    
    # 2) Apply respiratory bandpass filter (0.1-0.5 Hz → ~6-30 breaths/min)
    resp_filt = pre.bandpass_filter(cleaned, mode='resp')
    if resp_filt is None:
        return RespiratoryPreprocessResponse(success=False, peaks_count=0, error="Respiratory bandpass failed")
    
    # 3) Enhance peaks for better detection
    enhanced = pre.peak_enhancement(resp_filt)
    
    # 4) Remove outliers with Hampel filter
    denoised = pre.hampel_filter(enhanced)
    
    # 5) Compute global ESQI for quality metrics (no segmentation/filtering)
    entropy, esqi = pre.compute_esqi(denoised)
    
    # 6) Detect peaks in heart-rate band to extract IBIs (reuse same preprocessor instance)
    heart_band = pre.bandpass_filter(cleaned, mode='heart')
    peaks = pre.detect_peaks_dynamic(heart_band) if heart_band is not None else np.array([])
    ibi_t, ibi = pre.compute_ibis(peaks, pre.sampling_rate)

    # 7a) IBI-based PSD analysis (preferred method if sufficient IBIs available)
    method_used = "ibi-welch"
    freqs = np.array([])
    psd = np.array([])
    resp_rate_bpm: Optional[float] = None
    resp_freq_hz: Optional[float] = None
    peak_prom: Optional[float] = None
    dom_ratio: Optional[float] = None
    used_fallback = False

    if len(ibi) >= 8:
        # Resample IBI time series to uniform sampling rate
        t_even, ibi_even, fs_even = pre.resample_series(ibi_t, ibi, pre.ibi_resample_hz)
        
        # Compute power spectral density via Welch's method
        freqs, psd = pre.welch_psd(ibi_even - np.mean(ibi_even), fs_even)
        
        if freqs.size and psd.size:
            # Select respiratory peak in PSD within expected frequency band
            rr_bpm, rr_hz, prom, dom = pre.select_resp_peak(
                freqs,
                psd,
                band=(0.15, 0.5),
                min_bpm=9.0,
                max_bpm=60.0,
                min_prom_rel=0.12,
                dominance_ratio=0.9,
            )
            resp_rate_bpm, resp_freq_hz = rr_bpm, rr_hz
            peak_prom, dom_ratio = prom, dom

    # 7b) Fallback: signal-based PSD if IBI-based method failed
    if resp_rate_bpm is None:
        method_used = "resp-signal-welch"
        used_fallback = True
        base = denoised  # Use full denoised signal
        freqs, psd = pre.welch_psd(base - np.mean(base), pre.sampling_rate)
        if freqs.size and psd.size:
            rr_bpm, rr_hz, prom, dom = pre.select_resp_peak(
                freqs,
                psd,
                band=(0.15, 0.75),
                min_bpm=9.0,
                max_bpm=60.0,
                min_prom_rel=0.12,
                dominance_ratio=0.9,
            )
            resp_rate_bpm, resp_freq_hz = rr_bpm, rr_hz
            peak_prom, dom_ratio = prom, dom

    # 7c) Last-chance fallback: naive argmax on PSD if previous methods failed
    if resp_rate_bpm is None and freqs.size and psd.size:
        naive_bpm, naive_hz = pre.resp_rate_from_psd(freqs, psd, band=(0.15, 0.75))
        resp_rate_bpm = resp_rate_bpm or naive_bpm
        resp_freq_hz = resp_freq_hz or naive_hz

    if resp_rate_bpm is None:
        return RespiratoryPreprocessResponse(
            success=False,
            peaks_count=int(len(peaks)),
            esqi=float(esqi),
            entropy=float(entropy),
            error="Unable to estimate respiratory rate"
        )

    # Trim PSD vectors to prevent excessive payload size (limit to 2048 points)
    MAX_PSD_POINTS = 2048
    if freqs.size > MAX_PSD_POINTS:
        idx = np.linspace(0, freqs.size - 1, MAX_PSD_POINTS).astype(int)
        freqs_out = freqs[idx].tolist()
        psd_out = psd[idx].tolist()
    else:
        freqs_out = freqs.tolist()
        psd_out = psd.tolist()

    # Additionally, compute envelope-based PSD for diagnostics/validation
    analytic = hilbert(cleaned)  # ← Use raw PPG, not resp_filt
    env = np.abs(analytic)
    # Apply respiratory bandpass to envelope to isolate breathing modulation
    env_filt = pre.bandpass_filter(env, lowcut=0.15, highcut=0.5, mode='resp')
    if env_filt is None:
        env_filt = env  # Fallback if filter fails
    f_env, P_env = welch(env_filt - np.mean(env_filt), fs=pre.sampling_rate, nperseg=min(512, len(env_filt)))
    mask_env = (f_env >= 0.15) & (f_env <= 0.75)
    env_peak_hz = None
    env_peak_bpm = None
    if np.any(mask_env):
        idx_env = np.argmax(P_env[mask_env])
        env_peak_hz = f_env[mask_env][idx_env]
        env_peak_bpm = env_peak_hz * 60.0
        logger.info("Envelope-PSD peak: %.3f Hz (%.1f bpm), mag=%.3e", env_peak_hz, env_peak_bpm, P_env[mask_env][idx_env])

    # Log top IBI-PSD peaks for comparison
    if freqs.size and psd.size:
        idx_sorted = np.argsort(psd)[-6:][::-1]
        logger.info("Top IBI-PSD peaks:")
        for i in idx_sorted:
            logger.info("  %.4f Hz (%.1f bpm), mag=%.3e", freqs[i], freqs[i]*60, psd[i])
    
    # Log which method was used and both estimates
    if resp_rate_bpm is not None:
        logger.info("Final RR estimate: %.1f bpm (%.3f Hz) via %s", resp_rate_bpm, resp_freq_hz or 0, method_used)
        if env_peak_bpm is not None:
            diff = abs(resp_rate_bpm - env_peak_bpm)
            logger.info("Envelope vs Selected difference: %.1f bpm", diff)
    return RespiratoryPreprocessResponse(
        success=True,
        resp_rate_bpm=float(resp_rate_bpm),
        resp_freq_hz=float(resp_freq_hz) if resp_freq_hz is not None else None,
        esqi=float(esqi),
        entropy=float(entropy),
        peaks_count=int(len(peaks)),
        method_used=method_used,
        peak_prominence=peak_prom,
        dominance_ratio=dom_ratio,
        used_fallback=used_fallback,
        freqs=freqs_out,
        psd=psd_out,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)