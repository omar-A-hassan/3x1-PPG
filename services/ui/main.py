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
ENABLE_CSV_FALLBACK = True  # Download CSV if raw_signal missing from response
MOCK_AUTH_SERVICE = True    # Set to True to test login without auth service
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

@app.post("/update_result")
async def update_result(request: UpdateResultRequest):
    """
    Receiver service calls this after processing complete
    Now includes csv_file path, optional respiratory metrics, and in-memory signal/PSD data
    """
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
        )
        # Best-effort: if a CSV path was provided but the UI can't access it locally,
        # attempt to download it from the receiver service using the filename.
        # Only try if download endpoints are likely enabled (check for file existence first)
        if request.csv_file:
            try:
                csv_path = Path(request.csv_file)
                if not csv_path.exists():
                    # File doesn't exist locally - only attempt download if we suspect endpoints are enabled
                    # To avoid 404 errors, we skip download if ENABLE_DOWNLOAD_ENDPOINTS is known to be False
                    # For now, log the missing file but don't attempt download
                    logger.info(f"CSV file not found locally: {csv_path}. Skipping download (receiver may have downloads disabled).")
                    # If you enable receiver download endpoints, uncomment the code below:
                    # filename = csv_path.name
                    # download_url = f"{RECEIVER_SERVICE_URL}/download/{filename}"
                    # logger.info(f"Attempting to download from receiver: {download_url}")
                    # async with httpx.AsyncClient(timeout=30.0) as client:
                    #     resp = await client.get(download_url)
                    #     if resp.status_code == 200:
                    #         PPG_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    #         dest = PPG_DATA_DIR / filename
                    #         with open(dest, "wb") as f:
                    #             f.write(resp.content)
                    #         with result_state.lock:
                    #             result_state.latest_csv_file = str(dest)
                    #         logger.info(f"Downloaded CSV from receiver to {dest}")
                    #     else:
                    #         logger.warning(f"Failed to download CSV: {resp.status_code}")
                else:
                    # Path exists locally on UI host — store normalized path
                    with result_state.lock:
                        result_state.latest_csv_file = str(csv_path)
            except Exception as e:
                logger.warning(f"Error while checking CSV file for plotting: {e}")
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
    collection_status = status_data.get("status", "Unknown")
    
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
        if not username or not password:
            return "❌ Please enter username and password", ""
        
        # MOCK MODE: For testing without auth service
        if MOCK_AUTH_SERVICE:
            # Accept any username/password, generate fake API key
            import random
            import string
            fake_key = ''.join(random.choices(string.ascii_uppercase, k=3)) + str(random.randint(10, 99))
            auth_state.login(username, fake_key)
            logger.info(f"[MOCK] Login successful for {username}, API key: {fake_key}")
            return f"✅ Logged in as **{username}** (MOCK MODE)", fake_key
        
        # REAL MODE: Call auth service
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{AUTH_SERVICE_URL}/login",
                    json={"username": username, "password": password}
                )
                data = resp.json()
                
                if resp.status_code == 200 and data.get("success"):
                    auth_state.login(username, data["api_key"])
                    return f"✅ Logged in as **{username}**", data["api_key"]
                else:
                    return f"❌ {data.get('error', 'Login failed')}", ""
        except httpx.ConnectError:
            return "❌ Auth service not reachable", ""
        except Exception as e:
            logger.error(f"Login error: {e}")
            return f"❌ Error: {e}", ""

    def handle_logout():
        """Clear login session"""
        auth_state.logout()
        return "🔒 Logged out", ""

    def handle_regenerate_key(username: str, password: str):
        """Request new API key from auth service"""
        if not auth_state.is_logged_in():
            return "❌ Please login first", ""
        
        # MOCK MODE: Generate new fake key
        if MOCK_AUTH_SERVICE:
            import random
            import string
            new_key = ''.join(random.choices(string.ascii_uppercase, k=3)) + str(random.randint(10, 99))
            auth_state.api_key = new_key
            logger.info(f"[MOCK] Regenerated API key: {new_key}")
            return f"✅ New API key generated (MOCK MODE)", new_key
        
        # REAL MODE: Call auth service
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{AUTH_SERVICE_URL}/regenerate-api-key",
                    json={"username": username, "password": password}
                )
                data = resp.json()
                
                if resp.status_code == 200 and data.get("success"):
                    new_key = data["new_api_key"]
                    auth_state.api_key = new_key
                    return f"✅ New API key generated", new_key
                else:
                    return f"❌ {data.get('error', 'Failed')}", auth_state.api_key or ""
        except Exception as e:
            logger.error(f"Regenerate key error: {e}")
            return f"❌ Error: {e}", auth_state.api_key or ""

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
                create_placeholder_figure("No history to plot"),
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
        
        if glucose is None:
            return (
                status_markdown,
                "No prediction yet",
                "Waiting for model result...",
                "",
                _get_history_text(),
                create_placeholder_figure("No history to plot"),
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

- **< 70 mg/dL:** Hypoglycemia (Low) 🔴
- **70-140 mg/dL:** Normal 🟢
- **141-200 mg/dL:** Elevated 🟡
- **> 200 mg/dL:** Hyperglycemia (High) 🔴
"""
        return (status_markdown, display, details, ranges, _get_history_text(), _get_history_plot(), resp_info)

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

    @safe_plot("Failed to generate history plot")
    def _get_history_plot():
        """Plot glucose history"""
        history = result_state.get_history()
        if len(history) < 2:
            return create_placeholder_figure("Not enough history to plot")
        
        # Filter out None glucose values
        valid_history = [(i, r['glucose']) for i, r in enumerate(history) if r.get('glucose') is not None]
        
        if len(valid_history) < 2:
            return create_placeholder_figure("Not enough valid glucose readings to plot")
        
        indices, glucose_values = zip(*valid_history)
        
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        ax.plot(indices, glucose_values, 'bo-', 
               linewidth=2, markersize=8)
        ax.axhline(y=70, color='r', linestyle='--', label='Low threshold')
        ax.axhline(y=140, color='g', linestyle='--', label='Normal threshold')
        ax.axhline(y=200, color='orange', linestyle='--', label='High threshold')
        
        ax.set_xlabel('Measurement', fontsize=11)
        ax.set_ylabel('Glucose (mg/dL)', fontsize=11)
        ax.set_title('Glucose Level History', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig

    def get_signal_data():
        """Get signal data from in-memory result or CSV fallback.

        Returns: (timestamps[np.ndarray], ir_values[np.ndarray], data_source[str])
                 or (None, None, None) if unavailable
        """
        result = result_state.get_latest()
        if result and result.get("raw_signal") is not None and result.get("timestamps_ms") is not None:
            ir_values = np.array(result["raw_signal"], dtype=float)
            timestamps = np.array(result["timestamps_ms"], dtype=float) / 1000.0
            logger.info("Using in-memory raw signal for plotting")
            return timestamps, ir_values, "in-memory"
        if ENABLE_CSV_FALLBACK:
            csv_file = result_state.get_latest_csv()
            if csv_file and Path(csv_file).exists():
                t, y = load_ppg_from_csv(Path(csv_file))
                if t is not None and y is not None:
                    logger.info(f"Using CSV fallback for plotting: {csv_file}")
                    return np.asarray(t), np.asarray(y), f"CSV ({Path(csv_file).name})"
        logger.warning("No signal data available for plotting")
        return None, None, None

    def generate_signal_plots():
        timestamps, ir_values, data_source = get_signal_data()
        
        # If no data available, return placeholders
        if timestamps is None or ir_values is None:
            placeholders = [create_placeholder_figure("No data available")] * 9
            return (*placeholders, "⚠️ No data available. Please complete a collection first.")

        # Create plots
        overview = create_overview_plot(timestamps, ir_values)
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
        return (overview, *segments, spectrum, spectrogram, status_msg)

    def handle_respiratory_analysis(recompute: bool):
        """
        Unified respiratory analysis handler.

        - When recompute=False: display from in-memory data (fast refresh)
        - When recompute=True: recompute from latest CSV via preprocessing service

        Returns: (status_markdown: str, psd_plot: Figure)
        """
        if not recompute:
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

        # Recompute path: call preprocessing service using latest CSV
        csv_file = result_state.get_latest_csv()
        if not csv_file or not Path(csv_file).exists():
            return ("⚠️ No data file available", create_placeholder_figure("No data available"))

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resp_data = loop.run_until_complete(compute_respiratory_rate(Path(csv_file)))
            loop.close()
        except Exception as e:
            logger.error(f"Error in respiratory computation: {e}", exc_info=True)
            return (f"❌ Error: {e}", create_placeholder_figure("Computation failed"))

        if resp_data is None or not resp_data.get("success"):
            error_msg = resp_data.get("error", "Unknown error") if resp_data else "Request failed"
            return (f"⚠️ Analysis failed: {error_msg}", create_placeholder_figure("Analysis failed"))

        # Update result state with respiratory metrics (preserve other fields)
        result = result_state.get_latest() or {}
        result_state.update(
            glucose=result.get("glucose"),
            num_segments=result.get("num_segments", 0),
            quality_score=result.get("quality_score", 0.0),
            device=result.get("device", "Unknown"),
            csv_file=csv_file,
            resp_rate_bpm=resp_data.get("resp_rate_bpm"),
            resp_freq_hz=resp_data.get("resp_freq_hz"),
            esqi=resp_data.get("esqi"),
            entropy=resp_data.get("entropy"),
            peaks_count=resp_data.get("peaks_count"),
            method_used=resp_data.get("method_used"),
            raw_signal=result.get("raw_signal"),
            timestamps_ms=result.get("timestamps_ms"),
            freqs=resp_data.get("freqs"),
            psd=resp_data.get("psd"),
        )

        # Build plot from recomputed PSD
        freqs = resp_data.get("freqs", [])
        psd = resp_data.get("psd", [])
        resp_freq_hz = resp_data.get("resp_freq_hz")
        psd_plot = (
            create_respiratory_psd_plot(freqs, psd, resp_freq_hz)
            if freqs and psd else create_placeholder_figure("No PSD data available")
        )

        status = f"""
✅ **Respiratory Analysis Complete**

**Rate:** {resp_data.get('resp_rate_bpm', 0):.1f} breaths/min  
**Frequency:** {resp_data.get('resp_freq_hz', 0):.3f} Hz  
**ESQI:** {resp_data.get('esqi', 0):.3f}  
**Entropy:** {resp_data.get('entropy', 0):.3f}  
**Peaks:** {resp_data.get('peaks_count', 0)}  
**Method:** {resp_data.get('method_used', 'N/A')}
"""

        return (status, psd_plot)

    # ========================================================================
    # BUILD GRADIO INTERFACE
    # ========================================================================
    
    with gr.Blocks(title="PPG Glucose Monitor") as interface:
        with gr.Tab("🔐 Login"):
            gr.Markdown("## User Authentication")
            gr.Markdown("Login to get your API key for ESP32 configuration")
            
            with gr.Row():
                with gr.Column(scale=1):
                    username_input = gr.Textbox(label="Username", placeholder="Enter username")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                    
                    with gr.Row():
                        login_btn = gr.Button("🔑 Login", variant="primary")
                        logout_btn = gr.Button("🚪 Logout", variant="secondary")
            
            login_status = gr.Markdown("Not logged in")
            
            gr.Markdown("---")
            gr.Markdown("## Your API Key")
            gr.Markdown("*Copy this key to your ESP32 configuration*")
            
            api_key_display = gr.Textbox(label="API Key", value="", interactive=False)
            regenerate_btn = gr.Button("🔄 Generate New Key")
            
            # Connect buttons to functions
            login_btn.click(
                fn=handle_login,
                inputs=[username_input, password_input],
                outputs=[login_status, api_key_display]
            )
            
            logout_btn.click(
                fn=handle_logout,
                outputs=[login_status, api_key_display]
            )
            
            regenerate_btn.click(
                fn=handle_regenerate_key,
                inputs=[username_input, password_input],
                outputs=[login_status, api_key_display]
            )
        gr.Markdown("---")
        gr.Markdown("# 🩸 PPG-Based Glucose Monitor")
        gr.Markdown("")
        
        # ====================================================================
        # TAB 1: GLUCOSE MONITORING
        # ====================================================================
        
        with gr.Tab("📊 Glucose Monitor"):
            gr.Markdown("## Collection Status")
            gr.Markdown("*ESP32 runs autonomously - place finger on sensor to start collection*")
            
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
            gr.Markdown("## History")
            
            with gr.Row():
                history_text = gr.Markdown("No history yet")

            with gr.Row():
                history_plot = gr.Plot(label="Glucose Trends")

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
            gr.Markdown("### Complete Signal Overview")
            overview_plot = gr.Plot(label="Full PPG Signal")
            
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
            overview_plot,
            segment_plot_1,
            segment_plot_2,
            segment_plot_3,
            segment_plot_4,
            segment_plot_5,
            segment_plot_6,
            spectrum_plot,
            spectrogram_plot,
            plot_status
            ]
            )
        
        # ====================================================================
        # TAB 3: RESPIRATORY ANALYSIS
        # ====================================================================
        
        with gr.Tab("🫁 Respiratory Analysis"):
            gr.Markdown("## Respiratory Rate Estimation from PPG")
            gr.Markdown("Analyze inter-beat intervals using Welch's method to estimate respiratory rate")
            
            with gr.Row():
                recompute_checkbox = gr.Checkbox(label="Recompute from CSV", value=False)
                analyze_resp_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
            
            resp_status = gr.Markdown("Click 'Analyze' to show respiratory metrics (optionally recompute from CSV)")
            
            gr.Markdown("---")
            gr.Markdown("### Respiratory PSD Analysis")
            
            respiratory_psd_plot = gr.Plot(label="Respiratory Power Spectral Density")
            
            # Unified respiratory analysis binding
            analyze_resp_btn.click(
                fn=handle_respiratory_analysis,
                inputs=[recompute_checkbox],
                outputs=[resp_status, respiratory_psd_plot]
            )

        # ====================================================================
        # TAB 3: Blood Pressure Analysis
        # ====================================================================

        with gr.Tab("💢 Blood Pressure Monitor"):
            gr.Markdown("## This feature is under development")
            gr.Markdown("")

            with gr.Row():
                compute_bp_btn = gr.Button("🔎 Compute Blood Pressure", variant="primary", scale=1)

            gr.Markdown("###")
            gr.Markdown("---")
                

    return interface


# ============================================================================
# MAIN
# ============================================================================

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)