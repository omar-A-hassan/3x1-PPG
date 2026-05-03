"""
Enhanced UI Service with Plot Visualization
Displays glucose predictions AND signal plots after collection
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import logging
import httpx
import os
from typing import Optional, List
from datetime import datetime
import asyncio
from threading import Lock
from functools import wraps
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from scipy.signal import welch, spectrogram


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UI Service")

# ============================================================================
# DEBUG/DEPLOYMENT CONFIGURATION
# ============================================================================
ENABLE_CSV_FALLBACK = False  # Download CSV if raw_signal missing from response
MOCK_AUTH_SERVICE = False    # Set to True to test login without auth service
# ============================================================================

# Service URLs (configurable via environment variables)
# When running locally set RECEIVER_SERVICE_URL=http://localhost:8000
# When running under docker-compose set RECEIVER_SERVICE_URL=http://ble_receiver:8000
RECEIVER_SERVICE_URL = os.getenv("RECEIVER_SERVICE_URL", "http://localhost:8000")
PREPROCESSING_SERVICE_URL = os.getenv("PREPROCESSING_SERVICE_URL", "http://localhost:8001")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8004")

# Path to saved PPG data (matches receiver service)
PPG_DATA_DIR = Path("ppg_data")  # Adjust if receiver saves elsewhere


class UpdateResultRequest(BaseModel):
    glucose: Optional[float] = None
    num_segments: int
    quality_score: float
    device: str
    csv_file: Optional[str] = None  # Path to saved CSV
    # Respiratory metrics (optional)
    resp_rate_bpm: Optional[float] = None
    resp_freq_hz: Optional[float] = None
    esqi: Optional[float] = None
    entropy: Optional[float] = None
    peaks_count: Optional[int] = None
    method_used: Optional[str] = None
    # NEW: Raw signal and timestamps for in-memory plotting
    raw_signal: Optional[List[float]] = None
    timestamps_ms: Optional[List[float]] = None
    # PSD arrays for respiratory plot
    freqs: Optional[List[float]] = None
    psd: Optional[List[float]] = None
    # ECG data (parallel to PPG, uses same timestamps)
    ecg_signal: Optional[List[float]] = None


class ResultState:
    """Thread-safe container for results and data files"""

    def __init__(self):
        self.lock = Lock()
        self.latest_result = None
        self.history = []
        self.latest_csv_file = None  # NEW: Track latest data file

    def update(
        self,
        *,  # enforce keyword-only to prevent mistakes
        glucose: Optional[float] = None,
        num_segments: int = 0,
        quality_score: float = 0.0,
        device: str = "Unknown",
        csv_file: Optional[str] = None,
        resp_rate_bpm: Optional[float] = None,
        resp_freq_hz: Optional[float] = None,
        esqi: Optional[float] = None,
        entropy: Optional[float] = None,
        peaks_count: Optional[int] = None,
        method_used: Optional[str] = None,
        raw_signal: Optional[List[float]] = None,
        timestamps_ms: Optional[List[float]] = None,
        freqs: Optional[List[float]] = None,
        psd: Optional[List[float]] = None,
        ecg_signal: Optional[List[float]] = None,
    ) -> None:
        """Update result state with flexible keyword args.

        Preserves previous values when new ones are not provided (None).
        Appends to history (max 50 entries).
        """
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prev = self.latest_result or {}

            # Helper to keep previous value if new is None
            def keep(prev_key: str, new_val):
                return new_val if new_val is not None else prev.get(prev_key)

            result = {
                "timestamp": timestamp,
                "glucose": keep("glucose", glucose),
                "num_segments": keep("num_segments", num_segments),
                "quality_score": keep("quality_score", quality_score),
                "device": keep("device", device),
                "csv_file": keep("csv_file", csv_file),
                # Respiratory
                "resp_rate_bpm": keep("resp_rate_bpm", resp_rate_bpm),
                "resp_freq_hz": keep("resp_freq_hz", resp_freq_hz),
                "esqi": keep("esqi", esqi),
                "entropy": keep("entropy", entropy),
                "peaks_count": keep("peaks_count", peaks_count),
                "method_used": keep("method_used", method_used),
                # Data for plotting
                "raw_signal": keep("raw_signal", raw_signal),
                "timestamps_ms": keep("timestamps_ms", timestamps_ms),
                "freqs": keep("freqs", freqs),
                "psd": keep("psd", psd),
                # ECG data (uses same timestamps as PPG)
                "ecg_signal": keep("ecg_signal", ecg_signal),
            }

            self.latest_result = result
            self.latest_csv_file = result.get("csv_file")

            self.history.append(result)
            if len(self.history) > 50:
                self.history = self.history[-50:]

            # Logging summaries
            if result.get("glucose") is not None:
                try:
                    logger.info(
                        f"Updated: {float(result['glucose']):.1f} mg/dL, CSV: {self.latest_csv_file}"
                    )
                except Exception:
                    logger.info("Updated: glucose value present (formatting failed)")
            if result.get("resp_rate_bpm") is not None:
                logger.info(
                    f"Respiratory: {result['resp_rate_bpm']:.1f} bpm, ESQI: {result.get('esqi')}"
                )
            logger.debug("ResultState.update finished")

    def get_latest(self):
        with self.lock:
            return self.latest_result

    def get_latest_csv(self):
        with self.lock:
            return self.latest_csv_file

    def get_history(self):
        with self.lock:
            return self.history.copy()


result_state = ResultState()

class AuthState:
    """Track logged-in user session"""
    def __init__(self):
        self.username = None
        self.api_key = None
    
    def login(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key
    
    def logout(self):
        self.username = None
        self.api_key = None
    
    def is_logged_in(self) -> bool:
        return self.username is not None

auth_state = AuthState()

# endpoint that will return current user login status
# if login status is not logged in, return Logged out as False
@app.post("/login_status")
async def login_status():
    if auth_state.is_logged_in():
        return {"status": "logged_in", "username": auth_state.username, "api_key": auth_state.api_key}
    else:
        return {"status": "logged_out"}

@app.post("/update_result")
async def update_result(request: UpdateResultRequest):
    """
    Receiver service calls this after processing complete (PPG and ECG)
     to update latest glucose result and respiratory data."""
    try:
        result_state.update(
            glucose=request.glucose,
            num_segments=request.num_segments,
            quality_score=request.quality_score,
            device=request.device,
            csv_file=request.csv_file,
            resp_rate_bpm=request.resp_rate_bpm,
            resp_freq_hz=request.resp_freq_hz,
            esqi=request.esqi,
            entropy=request.entropy,
            peaks_count=request.peaks_count,
            method_used=request.method_used,
            raw_signal=request.raw_signal,
            timestamps_ms=request.timestamps_ms,
            freqs=request.freqs,
            psd=request.psd,
            ecg_signal=request.ecg_signal,
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to update result: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/health")
async def health():
    return {"service": "UI Service", "status": "ready"}


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def safe_plot(fallback_message: str = "Plot generation failed"):
    """Decorator to handle plot errors gracefully with full traceback logging.
    
    Args:
        fallback_message: Message to display in placeholder figure if plot fails
        
    Returns:
        Decorator function that wraps plotting functions with error handling
        
    Example:
        @safe_plot("Failed to create overview plot")
        def create_overview_plot(timestamps, ir_values):
            # ... plot code ...
            return fig
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"{func.__name__} failed: {e}",
                    exc_info=True  # ✅ Logs full stack trace for debugging
                )
                return create_placeholder_figure(fallback_message)
        return wrapper
    return decorator


# ============================================================================
# PLOTTING FUNCTIONS (from your data_plot.py)
# ============================================================================

def load_ppg_from_csv(csv_path: Path):
    """Load PPG data from CSV file"""
    try:
        logger.info(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Extract data
        timestamps = df['Timestamp_ms'].to_numpy(dtype=float) / 1000.0  # Convert to seconds
        ir_values = df['IR_Value'].to_numpy(dtype=float)

        logger.info(f"Loaded {len(ir_values)} samples")
        return timestamps, ir_values
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return None, None


def create_placeholder_figure(message: str, figsize=(10, 3)):
    """Create a simple matplotlib Figure containing a centered message.

    This is used as a safe fallback so Gradio never receives None for a Plot output.
    """
    try:
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    except Exception:
        # As a last resort, return an empty figure object
        return Figure(figsize=figsize)

@safe_plot("Failed to generate segment plots")
def create_segment_plots(timestamps, ir_values, segment_length=10.0):
    """
    Create time-domain segment plots
    Returns list of matplotlib figures (up to 6)
    """
    figures = []

    if len(timestamps) == 0:
        return figures

    start = float(np.min(timestamps))
    end = float(np.max(timestamps))

    # Create window edges; ensure last window covers the end (inclusive)
    edges = np.arange(start, end + segment_length, segment_length)
    if len(edges) <= 1:
        edges = np.array([start, start + segment_length])

    max_windows = min(len(edges) - 1, 6)

    for i in range(max_windows):
        seg_start = edges[i]
        seg_end = edges[i + 1]
        # Make the final window inclusive on the right edge so it captures the tail
        if i < max_windows - 1:
            mask = (timestamps >= seg_start) & (timestamps < seg_end)
        else:
            mask = (timestamps >= seg_start) & (timestamps <= seg_end)

        if mask.sum() == 0:
            continue

        fig = Figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        x = timestamps[mask]
        y = ir_values[mask]
        ax.plot(x, y, color='tab:blue', linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("IR Value", fontsize=10)
        ax.set_title(f"Segment {i+1}: {seg_start:.2f}s → {seg_end:.2f}s", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        figures.append(fig)

    return figures


@safe_plot("Failed to generate spectrum plot")
def create_spectrum_plot(ir_values, fs=100):
    """Create Welch power spectrum plot"""
    f, Pxx = welch(ir_values, fs=fs, nperseg=min(2048, len(ir_values)))
    
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    ax.semilogy(f, Pxx, linewidth=2)
    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Power Spectral Density', fontsize=11)
    ax.set_title('Welch Power Spectrum of IR Signal', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight physiological frequency range (0.5-3 Hz for heart rate)
    ax.axvspan(0.5, 3.0, alpha=0.2, color='green', label='Heart Rate Range')
    ax.legend()
    
    return fig


@safe_plot("Failed to generate spectrogram")
def create_spectrogram_plot(ir_values, fs=100):
    """Create spectrogram plot"""
    f, t, Sxx = spectrogram(ir_values, fs=fs, nperseg=256, noverlap=200)
    
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    im = ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency [Hz]', fontsize=11)
    ax.set_xlabel('Time [seconds]', fontsize=11)
    ax.set_title('Spectrogram of IR Signal', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Power [dB]')
    
    return fig


@safe_plot("Failed to generate respiratory PSD plot")
def create_respiratory_psd_plot(freqs, psd, resp_freq_hz=None):
    """Create respiratory PSD plot with highlighted peak"""
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    ax.plot(freqs, psd, linewidth=2, color='tab:blue')
    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Power Spectral Density', fontsize=11)
    ax.set_title('Respiratory Component PSD (Welch Method)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight respiratory band (0.1-0.4 Hz)
    ax.axvspan(0.1, 0.6, alpha=0.2, color='green', label='Respiratory Band')
    
    # Mark detected respiratory frequency
    if resp_freq_hz is not None and len(freqs) > 0:
        ax.axvline(x=resp_freq_hz, color='red', linestyle='--', linewidth=2, 
                  label=f'Detected: {resp_freq_hz:.3f} Hz')
    
    ax.legend()
    # Limit x-axis to 0-2 Hz for respiratory analysis
    ax.set_xlim(0, 1.0)
    
    return fig


async def compute_respiratory_rate(csv_path: Path):
    """Call preprocessing service to estimate respiratory rate from PPG data"""
    try:
        # Load IR values from CSV using helper
        _, ir_values = load_ppg_from_csv(csv_path)
        if ir_values is None:
            logger.error(f"Failed to load CSV data from {csv_path}")
            return None
        
        logger.info(f"Requesting respiratory analysis for {len(ir_values)} samples")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{PREPROCESSING_SERVICE_URL}/preprocess_respiratory",
                json={"signal": ir_values, "sampling_rate": 50}
            )
            
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"Respiratory analysis complete: {data.get('resp_rate_bpm')} bpm")
                return data
            else:
                logger.error(f"Respiratory preprocessing failed: {resp.status_code} {resp.text}")
                return None
                
    except Exception as e:
        logger.error(f"Failed to compute respiratory rate: {e}")
        return None


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def fetch_receiver_status() -> dict:
    """Fetch current status from the receiver service.
    
    Uses synchronous httpx since Gradio callbacks are sync by default.
    Returns dict with status info or error message.
    """
    try:
        # Use synchronous client - simpler and works in Gradio callbacks
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{RECEIVER_SERVICE_URL}/status")
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"Failed to fetch status: {resp.status_code}")
                return {"collection_status": "Error", "error": f"HTTP {resp.status_code}"}
    except httpx.ConnectError:
        return {"collection_status": "Disconnected", "error": "Receiver service not reachable"}
    except httpx.TimeoutException:
        return {"collection_status": "Timeout", "error": "Request timed out"}
    except Exception as e:
        logger.error(f"Error fetching receiver status: {e}")
        return {"collection_status": "Error", "error": str(e)}


def format_status_markdown(status_data: dict) -> str:
    """Format receiver status data as Markdown for display."""
    # Receiver returns "status", not "collection_status"
    collection_status = status_data.get("status", "Unknown/ Disconnected")
    
    # Map status to emoji
    status_emoji = {
        "ESP32 Not Connected": "🔌",
        "WAITING": "👆(Place Your Finger)",
        "COLLECTING": "📡",
        "TRANSMITTING": "📤",
        "PROCESSING": "⚙️",
        "COMPLETE": "✅",
        "READY": "🟢",
        "Disconnected": "🔴",
        "Error": "❌",
        "Timeout": "⏱️",
    }.get(collection_status, "❓")
    
    # Build status line
    status_line = f"**Status:** {status_emoji} {collection_status}"
    
    
    return status_line


def create_gradio_interface():

    # ========================================================================
    # AUTH HANDLER FUNCTIONS
    # ========================================================================
    
    def handle_login(username: str, password: str):
        """Call auth service to login, or use mock mode for testing"""
        try:
            if not username or not password:
                return "❌ Please enter username and password", "", gr.update(visible=True), gr.update(visible=False)
        except httpx.ConnectError:
            return "❌ Invalid input", "", gr.update(visible=True) , gr.update(visible=False)
        
        # MOCK MODE: For testing without auth service
        if MOCK_AUTH_SERVICE:
            # Accept any username/password, generate fake API key
            import random
            import string
            fake_key = ''.join(random.choices(string.ascii_uppercase, k=3)) + str(random.randint(10, 99))
            auth_state.login(username, fake_key)
            logger.info(f"[MOCK] Login successful for {username}, API key: {fake_key}")
            return f"✅ Logged in as {username} (MOCK MODE)", fake_key, gr.update(visible=False)
                    
            
        # REAL MODE: Call auth service
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{AUTH_SERVICE_URL}/login",
                    json={"username": username, "password": password}
                )
                data = resp.json()
                
                if resp.status_code == 200 and data.get("api_key"):
                    auth_state.login(username, data["api_key"])
                    return f"✅ Logged in as {username}", data["api_key"], gr.update(visible=False), gr.update(visible=True)
                else:
                    return f"❌ {data.get('detail', 'Login failed')}", "", gr.update(visible=True), gr.update(visible=False)
        except httpx.ConnectError:
            return "❌ Auth service not reachable", "", gr.update(visible=True), gr.update(visible=False)
        except Exception as e:
            logger.error(f"Login error: {e}")
            return f"❌ Error: {e}", "", gr.update(visible=True), gr.update(visible=False)

    def handle_logout():
        """Clear login session"""
        auth_state.logout()
        return "🔒 Logged out", "", gr.update(visible=True), gr.update(visible=False)

    def register_user(username: str, password: str):

        if not username or not password:
            return "❌ Please enter username and password"
        
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{AUTH_SERVICE_URL}/register",
                    json={"username": username, "password": password}
                )
                data = resp.json()
                
                if resp.status_code in [200,201] and data.get("status") == "success":
                    return f"✅ Registration successful for **{username}**, Please Login to receive API Key"
                else:
                    return f"❌ {data.get('detail', 'Registration failed')}"
                
        except httpx.ConnectError:
            return "❌ Auth service not reachable / Could not register successfully"
    # ========================================================================
    # GLUCOSE DISPLAY FUNCTIONS
    # ========================================================================

    def get_current_result():
        """Get latest glucose result and receiver status."""
        result = result_state.get_latest()
        
        # Fetch live status from receiver service
        status_data = fetch_receiver_status()
        status_markdown = format_status_markdown(status_data)
        
        if result is None:
            return (
                status_markdown,
                "No prediction yet",
                "Waiting for data...",
                "",
                _get_history_text(),
                create_placeholder_figure("No overview to plot"),
                "No respiratory data yet"
            )

        glucose = result.get("glucose")
        resp_rate = result.get("resp_rate_bpm")
        
        # Respiratory info
        resp_info = "**Respiratory Analysis:** Not yet computed"
        if resp_rate is not None:
            resp_info = f"""
**Respiratory Rate:** {resp_rate:.1f} breaths/min  
**Frequency:** {result.get('resp_freq_hz', 0):.3f} Hz  
**Signal Quality (ESQI):** {result.get('esqi', 0):.3f}  
**Entropy:** {result.get('entropy', 0):.3f}  
**Heart Peaks Detected:** {result.get('peaks_count', 0)}  
**Method:** {result.get('method_used', 'N/A')}
"""
        
        # Build overview plot from latest signal if available (independent of glucose)
        overview_fig = create_placeholder_figure("No overview to plot")
        try:
            timestamps, ir_values, source = get_signal_data()
            if timestamps is not None and ir_values is not None and len(ir_values) > 1:
                logger.info(f"Overview plot using {source}: {len(ir_values)} samples")
                overview_fig = create_overview_plot(timestamps, ir_values)
            else:
                logger.warning("Overview plot skipped: no signal data available")
        except Exception as e:
            logger.error(f"Overview plot failed: {e}", exc_info=True)
        
        if glucose is None:
            return (
                status_markdown,
                "No prediction yet",
                "Waiting for model result...",
                "",
                _get_history_text(),
                overview_fig,
                resp_info
            )
        if glucose < 70:
            status = "⚠️ LOW"
            color = "🔴"
        elif glucose <= 140:
            status = "✅ NORMAL"
            color = "🟢"
        elif glucose <= 200:
            status = "⚠️ ELEVATED"
            color = "🟡"
        else:
            status = "⚠️ HIGH"
            color = "🔴"

        display = f"{color} {glucose:.1f} mg/dL - {status}"
        details = f"""
**Timestamp:** {result['timestamp']}  
**Quality Score:** {result['quality_score']:.2%}  
**Segments Analyzed:** {result['num_segments']}  
**Device:** {result['device']}  
**Data File:** {result.get('csv_file', 'N/A')}
"""
        ranges = """
### Reference Ranges:

"""
        return (status_markdown, display, details, ranges, _get_history_text(), overview_fig, resp_info)

    def _get_history_text():
        """Get history as markdown text"""
        history = result_state.get_history()
        if not history:
            return "No history yet"
        lines = ["### Recent Predictions:\n"]
        for r in reversed(history[-10:]):
            glucose_val = r.get('glucose')
            if glucose_val is not None:
                glucose_str = f"{glucose_val:.1f} mg/dL"
            else:
                glucose_str = "N/A"
            lines.append(
                f"- **{r['timestamp']}**: {glucose_str} "
                f"(Quality: {r['quality_score']:.1%}, Segments: {r['num_segments']})"
            )
        return "\n".join(lines)

    @safe_plot("Failed to generate overview plot")
    def create_overview_plot(timestamps, ir_values):
        """Create overview plot of entire signal"""
        fig = Figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        
        ax.plot(timestamps, ir_values, linewidth=0.5, alpha=0.8, color='blue')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('IR Value', fontsize=11)
        
        # Dynamic title based on actual duration
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0
        ax.set_title(f'Complete PPG Signal ({duration:.0f} seconds)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f"Samples: {len(ir_values)}\n"
        stats_text += f"Mean: {ir_values.mean():.1f}\n"
        stats_text += f"Std: {ir_values.std():.1f}\n"
        stats_text += f"Range: [{ir_values.min():.0f}, {ir_values.max():.0f}]"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        return fig

    def get_signal_data_by_type(signal_type="PPG"):
        """Get signal data (PPG or ECG) from in-memory result or CSV fallback.

        Args:
            signal_type: "PPG" or "ECG"

        Returns: (timestamps[np.ndarray], signal_values[np.ndarray], data_source[str])
                 or (None, None, None) if unavailable
        """
        result = result_state.get_latest()
        
        # Determine field names based on signal type
        # Note: Both PPG and ECG use the same timestamps (collected synchronously)
        if signal_type == "ECG":
            signal_field = "ecg_signal"
            signal_label = "ECG"
        else:  # Default to PPG
            signal_field = "raw_signal"
            signal_label = "PPG"
        
        # Both use timestamps_ms (shared timestamps)
        if result and result.get(signal_field) is not None and result.get("timestamps_ms") is not None:
            signal_values = np.array(result[signal_field], dtype=float)
            timestamps = np.array(result["timestamps_ms"], dtype=float) / 1000.0
            logger.info(f"Using in-memory {signal_label} signal for plotting")
            return timestamps, signal_values, f"in-memory {signal_label}"
        
        if ENABLE_CSV_FALLBACK and signal_type == "PPG":  # CSV fallback only for PPG
            csv_file = result_state.get_latest_csv()
            if csv_file and Path(csv_file).exists():
                t, y = load_ppg_from_csv(Path(csv_file))
                if t is not None and y is not None:
                    logger.info(f"Using CSV fallback for plotting: {csv_file}")
                    return np.asarray(t), np.asarray(y), f"CSV ({Path(csv_file).name})"
        
        logger.warning(f"No {signal_label} signal data available for plotting")
        return None, None, None

    def get_signal_data():
        """Get PPG signal data (backward compatibility wrapper)"""
        return get_signal_data_by_type("PPG")

    def generate_signal_plots():
        timestamps, ir_values, data_source = get_signal_data()
        
        # If no data available, return placeholders
        if timestamps is None or ir_values is None:
            placeholders = [create_placeholder_figure("No data available")] * 8
            return (*placeholders, "⚠️ No data available. Please complete a collection first.")

        # Create plots
        segments = create_segment_plots(timestamps, ir_values, segment_length=10.0)

        # Pad segments to 6 items with placeholder figures
        while len(segments) < 6:
            segments.append(create_placeholder_figure("No segment data"))
        segments = segments[:6]

        # Estimate sampling rate from timestamps for frequency-domain plots
        diffs = np.diff(timestamps)
        fs = float(round(1.0 / np.median(diffs))) if len(diffs) > 0 and np.median(diffs) > 0 else 100.0

        spectrum = create_spectrum_plot(ir_values, fs=int(fs)) or create_placeholder_figure("No spectrum")
        spectrogram = create_spectrogram_plot(ir_values, fs=int(fs)) or create_placeholder_figure("No spectrogram")

        status_msg = f"✅ Generated plots from {data_source} (fs≈{int(fs)} Hz)"

        # **Unpack segments list into individual outputs**
        return (*segments, spectrum, spectrogram, status_msg)

    def generate_ecg_plots():
        """Generate ECG signal plots (reuses existing plotting functions)"""
        timestamps, ecg_values, data_source = get_signal_data_by_type("ECG")
        
        # If no data available, return placeholders
        if timestamps is None or ecg_values is None:
            placeholders = [create_placeholder_figure("No ECG data available")] * 8
            return (*placeholders, "⚠️ No ECG data available. Please complete a collection first.")

        # Create plots using same functions as PPG
        segments = create_segment_plots(timestamps, ecg_values, segment_length=10.0)

        # Pad segments to 6 items with placeholder figures
        while len(segments) < 6:
            segments.append(create_placeholder_figure("No segment data"))
        segments = segments[:6]

        # Estimate sampling rate
        diffs = np.diff(timestamps)
        fs = float(round(1.0 / np.median(diffs))) if len(diffs) > 0 and np.median(diffs) > 0 else 50.0

        spectrum = create_spectrum_plot(ecg_values, fs=int(fs)) or create_placeholder_figure("No spectrum")
        spectrogram = create_spectrogram_plot(ecg_values, fs=int(fs)) or create_placeholder_figure("No spectrogram")

        status_msg = f"✅ Generated ECG plots from {data_source} (fs≈{int(fs)} Hz)"

        return (*segments, spectrum, spectrogram, status_msg)

    def handle_respiratory_analysis():
        """
        Unified respiratory analysis handler.

        - When recompute=False: display from in-memory data (fast refresh)
        - When recompute=True: recompute from latest CSV via preprocessing service

        Returns: (status_markdown: str, psd_plot: Figure)
        """
        
        # Fast path: show in-memory PSD if available
        result = result_state.get_latest()
        if not result:
            return ("⚠️ No results available", create_placeholder_figure("No data available"))

        freqs = result.get("freqs")
        psd = result.get("psd")
        resp_freq_hz = result.get("resp_freq_hz")

        if freqs is not None and psd is not None:
            freqs_arr = np.array(freqs)
            psd_arr = np.array(psd)
            psd_plot = create_respiratory_psd_plot(freqs_arr, psd_arr, resp_freq_hz)

            resp_rate_bpm = result.get("resp_rate_bpm")
            esqi = result.get("esqi")
            entropy = result.get("entropy")
            peaks_count = result.get("peaks_count")
            method_used = result.get("method_used")

            if resp_rate_bpm is not None:
                status = f"""
✅ **Respiratory Metrics (from latest result)**

**Rate:** {resp_rate_bpm:.1f} breaths/min  
**Frequency:** {resp_freq_hz:.3f} Hz  
**ESQI:** {esqi:.3f}  
**Entropy:** {entropy:.3f}  
**Peaks:** {peaks_count}  
**Method:** {method_used}
"""
            else:
                status = "ℹ️ Respiratory data available (PSD plot shown)"

            return (status, psd_plot)

        return ("⚠️ No respiratory analysis data available", create_placeholder_figure("No PSD data"))


    # ========================================================================
    # BUILD GRADIO INTERFACE
    # ========================================================================
    
    with gr.Blocks(title="PPG Glucose Monitor") as interface:
        # ====================================================================
        gr.Markdown("---")
        gr.Markdown("# 🩸 PPG-Based Glucose Monitor")
        gr.Markdown("")

        # ====================================================================
        login_panel = gr.Group(elem_id="login-box", elem_classes=["login-card"])
        with login_panel:
            with gr.Tab("🔐 Login"):
                gr.Markdown("## User Authentication")
                gr.Markdown("Login to get your API key for ESP32 configuration")
                gr.Markdown("  \n")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        username_input = gr.Textbox(label="Username", placeholder="Enter username")
                        password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                    
                
                
                
            #Tab for registering new users, calls out the /register endpoint of auth service
            with gr.Tab("➕ Register New User"):
                gr.Markdown("## User Registration")
                gr.Markdown("---")
                gr.Markdown("This feature is not implemented in the UI at this time.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        username_input_2 = gr.Textbox(label="Username", placeholder="Enter New username")
                        password_input_2 = gr.Textbox(label="Password", type="password", placeholder="Enter New password")

                        with gr.Row():
                            register_btn = gr.Button("🆕 Register", variant="primary")

                register_status = gr.Markdown("Please Enter Credentials to Register")
                register_btn.click(fn=register_user,inputs=[username_input_2, password_input_2], outputs=[register_status])
        
        with gr.Row():
            with gr.Column(scale=1):
                status_display = gr.Textbox(value="Not logged in", label="Login Status", interactive=False)
            with gr.Column(scale=1):
                api_key_display = gr.Textbox(label="API Key", value="", interactive=False)
            with gr.Column(scale=1):
                login_btn = gr.Button("🔑 Login", variant="primary", size="lg")
                logout_btn = gr.Button("🔒 Logout", variant="secondary", size="lg")
                



        gr.Markdown("---")
        # ====================================================================
        # TAB 1: GLUCOSE MONITORING
        # ====================================================================
        
        data_panel = gr.Group(elem_id="Data-box", elem_classes=["login-card"],visible=False)
        with data_panel:
            
            with gr.Tab("📊 Glucose Monitor"):
                gr.Markdown("## Collection Status")
                gr.Markdown("*ESP32 runs autonomously - place finger on sensor to start collection*")
                gr.Markdown("___")
                
                with gr.Row():
                    status_box = gr.Markdown(value="**Status:** ⏸️ Refresh status...")
                    refresh_btn = gr.Button("🔄 Refresh Status", variant="primary", scale=1)

                gr.Markdown("---")
                gr.Markdown("## Current Prediction")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        glucose_output = gr.Markdown("### Waiting for data...")
                        details_output = gr.Markdown("No prediction yet")
                    
                    with gr.Column(scale=1):
                        ranges_output = gr.Markdown("")
                        respiratory_output = gr.Markdown("**Respiratory Analysis:** Not yet computed")

                gr.Markdown("---")
                gr.Markdown("## Overview")
                
                with gr.Row():
                    history_text = gr.Markdown("No history yet")

                with gr.Row():
                    history_plot = gr.Plot(label="PPG Overview")

                # Button bindings
                refresh_btn.click(
                    fn=get_current_result,
                    outputs=[status_box, glucose_output, details_output, ranges_output, history_text, history_plot, respiratory_output]
                )
            
            # ====================================================================
            # TAB 2: SIGNAL VISUALIZATION
            # ====================================================================
            
            with gr.Tab("📈 Signal Analysis"):
                gr.Markdown("## Raw PPG Signal Visualization")
                gr.Markdown("View time-domain and frequency-domain analysis of collected data (60 seconds)")
                
                with gr.Row():
                    plot_btn = gr.Button("🔄 Generate Plots", variant="primary", size="lg")
                
                plot_status = gr.Markdown("Click 'Generate Plots' to visualize the latest collection")
                
                gr.Markdown("---")
                gr.Markdown("### Time-Domain Segments (10-second windows)")
                
                with gr.Row():
                    segment_plot_1 = gr.Plot(label="Segment 1")
                    segment_plot_2 = gr.Plot(label="Segment 2")
                
                with gr.Row():
                    segment_plot_3 = gr.Plot(label="Segment 3")
                    segment_plot_4 = gr.Plot(label="Segment 4")
                
                with gr.Row():
                    segment_plot_5 = gr.Plot(label="Segment 5")
                    segment_plot_6 = gr.Plot(label="Segment 6")
                
                gr.Markdown("---")
                gr.Markdown("### Frequency-Domain Analysis")
                
                with gr.Row():
                    spectrum_plot = gr.Plot(label="Power Spectrum")
                
                with gr.Row():
                    spectrogram_plot = gr.Plot(label="Spectrogram")
                
                # Plot generation binding
                plot_btn.click(
                    fn=generate_signal_plots,
                    outputs=[
                        segment_plot_1,
                        segment_plot_2,
                        segment_plot_3,
                        segment_plot_4,
                        segment_plot_5,
                        segment_plot_6,
                        spectrum_plot,
                        spectrogram_plot,
                        plot_status,
                    ],
                )
            
            # ====================================================================
            # TAB 3: RESPIRATORY ANALYSIS
            # ====================================================================
            
            with gr.Tab("🫁 Respiratory Analysis"):
                gr.Markdown("## Respiratory Rate Estimation from PPG")
                gr.Markdown("Analyze inter-beat intervals using Welch's method to estimate respiratory rate")
                
                with gr.Row():
                    analyze_resp_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                resp_status = gr.Markdown("Click 'Analyze' to show respiratory metrics (optionally recompute from CSV)")
                
                gr.Markdown("---")
                gr.Markdown("### Respiratory PSD Analysis")
                
                respiratory_psd_plot = gr.Plot(label="Respiratory Power Spectral Density")
                
                # Unified respiratory analysis binding
                analyze_resp_btn.click(
                    fn=handle_respiratory_analysis,
                    inputs=[],
                    outputs=[resp_status, respiratory_psd_plot]
                )

            # ====================================================================
            # TAB 4: ECG ANALYSIS
            # ====================================================================
            
            with gr.Tab("⚡ ECG Analysis"):
                gr.Markdown("## ECG Signal Visualization")
                gr.Markdown("View time-domain and frequency-domain analysis of collected ECG data (60 seconds)")
                
                with gr.Row():
                    ecg_plot_btn = gr.Button("🔄 Generate ECG Plots", variant="primary", size="lg")
                
                ecg_plot_status = gr.Markdown("Click 'Generate ECG Plots' to visualize the latest collection")
                
                gr.Markdown("---")
                gr.Markdown("### Time-Domain Segments (10-second windows)")
                
                with gr.Row():
                    ecg_segment_plot_1 = gr.Plot(label="Segment 1")
                    ecg_segment_plot_2 = gr.Plot(label="Segment 2")
                
                with gr.Row():
                    ecg_segment_plot_3 = gr.Plot(label="Segment 3")
                    ecg_segment_plot_4 = gr.Plot(label="Segment 4")
                
                with gr.Row():
                    ecg_segment_plot_5 = gr.Plot(label="Segment 5")
                    ecg_segment_plot_6 = gr.Plot(label="Segment 6")
                
                gr.Markdown("---")
                gr.Markdown("### Frequency-Domain Analysis")
                
                with gr.Row():
                    ecg_spectrum_plot = gr.Plot(label="Power Spectrum")
                
                with gr.Row():
                    ecg_spectrogram_plot = gr.Plot(label="Spectrogram")
                
                # Plot generation binding
                ecg_plot_btn.click(
                    fn=generate_ecg_plots,
                    outputs=[
                        ecg_segment_plot_1,
                        ecg_segment_plot_2,
                        ecg_segment_plot_3,
                        ecg_segment_plot_4,
                        ecg_segment_plot_5,
                        ecg_segment_plot_6,
                        ecg_spectrum_plot,
                        ecg_spectrogram_plot,
                        ecg_plot_status,
                    ],
                )

            # ====================================================================
            # TAB 5: Blood Pressure Analysis
            # ====================================================================

            with gr.Tab("💢 Blood Pressure Monitor"):
                gr.Markdown("## This feature is under development")
                gr.Markdown("")

                with gr.Row():
                    compute_bp_btn = gr.Button("🔎 Compute Blood Pressure", variant="primary", scale=1)

                gr.Markdown("###")
                gr.Markdown("---")


        # ====================================================================
        # Connect buttons to functions using the global status_display

        login_btn.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[status_display, api_key_display, login_panel,data_panel]
        )

        logout_btn.click(
            fn=handle_logout,
            outputs=[status_display, api_key_display, login_panel,data_panel]
        )

        gr.HTML("""
<style>

    /* OUTER CARD */
    .login-card {
        border: 1px solid #444;
        border-radius: 20px !important;
        padding: 20px;
        background: #1e1e24;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }

    /* TEXTBOX + INPUT FIELDS */
    .login-card .gr-textbox input,
    .login-card .gr-textbox textarea,
    .login-card input,
    .login-card textarea {
        border-radius: 12px !important;
        padding: 10px 14px !important;
        background: #2a2a30 !important;
        border: 1px solid #555 !important;
        color: white !important;
    }

    /* BUTTONS */
    .login-card .gr-button > button,
    .login-card .gr-button,
    .login-card button {
        border-radius: 12px !important;
        padding: 10px 18px !important;
    }

    /* LABELS */
    .login-card label {
        font-size: 14px;
        font-weight: 600;
    }

</style>
""")

    return interface


# ============================================================================
# MAIN
# ============================================================================

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)