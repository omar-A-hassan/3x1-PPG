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
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.signal import welch, spectrogram


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UI Service")

# ============================================================================
# DEBUG/DEPLOYMENT CONFIGURATION
# ============================================================================
ENABLE_CSV_FALLBACK = False  # Download CSV if raw_signal missing from response
# ============================================================================

# Service URLs (configurable via environment variables)
# When running locally set RECEIVER_SERVICE_URL=http://localhost:8000
# When running under docker-compose set RECEIVER_SERVICE_URL=http://ble_receiver:8000
RECEIVER_SERVICE_URL = os.getenv("RECEIVER_SERVICE_URL", "http://localhost:8000")
PREPROCESSING_SERVICE_URL = os.getenv("PREPROCESSING_SERVICE_URL", "http://localhost:8001")

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

    def update(self, glucose: Optional[float], num_segments: int, quality_score: float, 
               device: str, csv_file: Optional[str] = None,
               resp_rate_bpm: Optional[float] = None,
               resp_freq_hz: Optional[float] = None,
               esqi: Optional[float] = None,
               entropy: Optional[float] = None,
               peaks_count: Optional[int] = None,
               method_used: Optional[str] = None,
               raw_signal: Optional[List[float]] = None,
               timestamps_ms: Optional[List[float]] = None,
               freqs: Optional[List[float]] = None,
               psd: Optional[List[float]] = None):
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Preserve previous glucose value if not provided (avoid overwriting model result)
            prev = self.latest_result or {}
            glucose_val = glucose if glucose is not None else prev.get("glucose")
            result = {
                "timestamp": timestamp,
                "glucose": glucose_val,
                "num_segments": num_segments,
                "quality_score": quality_score,
                "device": device,
                "csv_file": csv_file,
                "resp_rate_bpm": resp_rate_bpm,
                "resp_freq_hz": resp_freq_hz,
                "esqi": esqi,
                "entropy": entropy,
                "peaks_count": peaks_count,
                "method_used": method_used,
                "raw_signal": raw_signal,
                "timestamps_ms": timestamps_ms,
                "freqs": freqs,
                "psd": psd,
            }
            self.latest_result = result
            self.latest_csv_file = csv_file  # NEW
            self.history.append(result)
            if len(self.history) > 50:
                self.history = self.history[-50:]
            if glucose_val is not None:
                logger.info(f"Updated result: {float(glucose_val):.1f} mg/dL, CSV: {csv_file}")
            else:
                logger.info(f"Updated result: glucose=None, CSV: {csv_file}")
            if resp_rate_bpm is not None:
                logger.info(f"Respiratory rate: {resp_rate_bpm:.1f} bpm, ESQI: {esqi}")
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


@app.post("/update_result")
async def update_result(request: UpdateResultRequest):
    """
    Receiver service calls this after processing complete
    Now includes csv_file path, optional respiratory metrics, and in-memory signal/PSD data
    """
    try:
        result_state.update(
            request.glucose,
            request.num_segments,
            request.quality_score,
            request.device,
            request.csv_file,
            request.resp_rate_bpm,
            request.resp_freq_hz,
            request.esqi,
            request.entropy,
            request.peaks_count,
            request.method_used,
            request.raw_signal,
            request.timestamps_ms,
            request.freqs,
            request.psd,
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


def create_spectrum_plot(ir_values, fs=100):
    """Create Welch power spectrum plot"""
    try:
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
    except Exception as e:
        logger.error(f"Failed to create spectrum plot: {e}")
        return create_placeholder_figure("Failed to generate spectrum plot")


def create_spectrogram_plot(ir_values, fs=100):
    """Create spectrogram plot"""
    try:
        f, t, Sxx = spectrogram(ir_values, fs=fs, nperseg=256, noverlap=200)
        
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        im = ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency [Hz]', fontsize=11)
        ax.set_xlabel('Time [seconds]', fontsize=11)
        ax.set_title('Spectrogram of IR Signal', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Power [dB]')
        
        return fig
    except Exception as e:
        logger.error(f"Failed to create spectrogram: {e}")
        return create_placeholder_figure("Failed to generate spectrogram")


def create_respiratory_psd_plot(freqs, psd, resp_freq_hz=None):
    """Create respiratory PSD plot with highlighted peak"""
    try:
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        ax.plot(freqs, psd, linewidth=2, color='tab:blue')
        ax.set_xlabel('Frequency [Hz]', fontsize=11)
        ax.set_ylabel('Power Spectral Density', fontsize=11)
        ax.set_title('Respiratory Component PSD (Welch Method)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight respiratory band (0.1-0.4 Hz)
        ax.axvspan(0.1, 0.4, alpha=0.2, color='green', label='Respiratory Band')
        
        # Mark detected respiratory frequency
        if resp_freq_hz is not None and len(freqs) > 0:
            ax.axvline(x=resp_freq_hz, color='red', linestyle='--', linewidth=2, 
                      label=f'Detected: {resp_freq_hz:.3f} Hz')
        
        ax.legend()
        xlim_max = max(0.5, freqs[-1]) if len(freqs) > 0 else 0.5
        ax.set_xlim(0, xlim_max)
        
        return fig
    except Exception as e:
        logger.error(f"Failed to create respiratory PSD plot: {e}")
        return create_placeholder_figure("Failed to generate respiratory PSD plot")


def create_overview_plot(timestamps, ir_values):
    """Create overview plot of entire signal"""
    try:
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
    except Exception as e:
        logger.error(f"Failed to create overview plot: {e}")
        return create_placeholder_figure("Failed to generate overview plot")


async def compute_respiratory_rate(csv_path: Path):
    """Call preprocessing service to estimate respiratory rate from PPG data"""
    try:
        # Load IR values from CSV
        df = pd.read_csv(csv_path)
        ir_values = df['IR_Value'].tolist()
        
        logger.info(f"Requesting respiratory analysis for {len(ir_values)} samples")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{PREPROCESSING_SERVICE_URL}/preprocess_respiratory",
                json={"signal": ir_values, "sampling_rate": 100}
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

def create_gradio_interface():
    
    async def send_command(endpoint: str):
        """Send POST to /collect or /stop"""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(f"{RECEIVER_SERVICE_URL}/{endpoint}")
                if resp.status_code == 200:
                    return f"✅ {endpoint.capitalize()} command sent successfully."
                else:
                    return f"⚠️ Failed: {resp.text}"
        except Exception as e:
            return f"❌ Error: {e}"

    def send_command_sync(endpoint: str):
        """Synchronous helper for Gradio button callbacks to avoid asyncio.run nested loops.

        Uses httpx sync API for simplicity.
        """
        try:
            resp = httpx.post(f"{RECEIVER_SERVICE_URL}/{endpoint}", timeout=20.0)
            if resp.status_code == 200:
                return f"✅ {endpoint.capitalize()} command sent successfully."
            else:
                return f"⚠️ Failed: {resp.text}"
        except Exception as e:
            return f"❌ Error: {e}"

    def get_current_result():
        """Get latest glucose result"""
        result = result_state.get_latest()
        if result is None:
            return (
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
        return (display, details, ranges, _get_history_text(), _get_history_plot(), resp_info)

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

    def _get_history_plot():
        """Plot glucose history"""
        try:
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
        except Exception as e:
            logger.error(f"Failed to plot history: {e}")
            return create_placeholder_figure("Failed to generate history plot")

    def generate_signal_plots():
        result = result_state.get_latest()
        
        # Try to use in-memory data first
        timestamps = None
        ir_values = None
        data_source = None
        
        if result and result.get("raw_signal") is not None and result.get("timestamps_ms") is not None:
            # Use in-memory data
            ir_values = np.array(result["raw_signal"])
            timestamps = np.array(result["timestamps_ms"]) / 1000.0  # Convert ms to seconds
            data_source = "in-memory"
            logger.info("Using in-memory raw signal for plotting")
        elif ENABLE_CSV_FALLBACK:
            # Fallback to CSV file
            csv_file = result_state.get_latest_csv()
            if csv_file and Path(csv_file).exists():
                timestamps, ir_values = load_ppg_from_csv(Path(csv_file))
                if timestamps is not None:
                    data_source = f"CSV ({Path(csv_file).name})"
                    logger.info(f"Using CSV fallback for plotting: {csv_file}")
        
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

    def compute_respiratory_sync():
        """Synchronous wrapper to compute respiratory rate and update result state"""
        csv_file = result_state.get_latest_csv()
        if not csv_file or not Path(csv_file).exists():
            return ("⚠️ No data file available", create_placeholder_figure("No data available"))
        
        # Run async function in new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resp_data = loop.run_until_complete(compute_respiratory_rate(Path(csv_file)))
            loop.close()
        except Exception as e:
            logger.error(f"Error in respiratory computation: {e}")
            return (f"❌ Error: {e}", create_placeholder_figure("Computation failed"))
        
        if resp_data is None or not resp_data.get("success"):
            error_msg = resp_data.get("error", "Unknown error") if resp_data else "Request failed"
            return (f"⚠️ Analysis failed: {error_msg}", create_placeholder_figure("Analysis failed"))
        
        # Update result state with respiratory metrics
        result = result_state.get_latest()
        if result:
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
        
        # Create PSD plot
        freqs = resp_data.get("freqs", [])
        psd = resp_data.get("psd", [])
        resp_freq_hz = resp_data.get("resp_freq_hz")
        
        if freqs and psd:
            psd_plot = create_respiratory_psd_plot(freqs, psd, resp_freq_hz)
        else:
            psd_plot = create_placeholder_figure("No PSD data available")
        
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

    def refresh_respiratory_display():
        """Display respiratory PSD from in-memory data (if available)"""
        result = result_state.get_latest()
        
        if not result:
            return ("⚠️ No results available", create_placeholder_figure("No data available"))
        
        # Check if we have in-memory PSD data
        freqs = result.get("freqs")
        psd = result.get("psd")
        resp_freq_hz = result.get("resp_freq_hz")
        
        if freqs is not None and psd is not None:
            # Convert lists to numpy arrays for plotting
            freqs_arr = np.array(freqs)
            psd_arr = np.array(psd)
            psd_plot = create_respiratory_psd_plot(freqs_arr, psd_arr, resp_freq_hz)
            
            # Build status message
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
        else:
            return ("⚠️ No respiratory analysis data available", create_placeholder_figure("No PSD data"))

    # ========================================================================
    # BUILD GRADIO INTERFACE
    # ========================================================================
    
    with gr.Blocks(title="PPG Glucose Monitor") as interface:
        gr.Markdown("# 🩸 PPG-Based Glucose Monitor")
        gr.Markdown("Real-time glucose prediction from photoplethysmography signals")
        
        # ====================================================================
        # TAB 1: GLUCOSE MONITORING
        # ====================================================================
        
        with gr.Tab("📊 Glucose Monitor"):
            gr.Markdown("## Collection Control")
            
            with gr.Row():
                start_btn = gr.Button("▶️ Start Collection", variant="primary", scale=1)
                stop_btn = gr.Button("⏹ Stop Collection", variant="secondary", scale=1)
                status_box = gr.Markdown(value="**Status:** Ready")
                refresh_btn = gr.Button("🔄 Refresh")

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

            # Button bindings (use synchronous helper to avoid nested event loops)
            start_btn.click(
                fn=lambda: send_command_sync("collect"),
                outputs=status_box,
            )
            stop_btn.click(
                fn=lambda: send_command_sync("stop"),
                outputs=status_box,
            )

            refresh_btn.click(
            fn=get_current_result,
                outputs=[glucose_output, details_output, ranges_output, history_text, history_plot, respiratory_output]
                )
        
        # ====================================================================
        # TAB 2: SIGNAL VISUALIZATION
        # ====================================================================
        
        with gr.Tab("📈 Signal Analysis"):
            gr.Markdown("## Raw PPG Signal Visualization")
            gr.Markdown("View time-domain and frequency-domain analysis of collected data (90 seconds)")
            
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
                refresh_resp_btn = gr.Button("🔄 Refresh Display", variant="secondary", size="lg")
                compute_resp_btn = gr.Button("🔬 Re-compute Respiratory", variant="primary", size="lg")
            
            resp_status = gr.Markdown("Click 'Refresh Display' to show latest results or 'Re-compute' to reanalyze from CSV")
            
            gr.Markdown("---")
            gr.Markdown("### Respiratory PSD Analysis")
            
            respiratory_psd_plot = gr.Plot(label="Respiratory Power Spectral Density")
            
            # Respiratory refresh binding (uses in-memory data)
            refresh_resp_btn.click(
                fn=refresh_respiratory_display,
                outputs=[resp_status, respiratory_psd_plot]
            )
            
            # Respiratory re-computation binding (recomputes from CSV)
            compute_resp_btn.click(
                fn=compute_respiratory_sync,
                outputs=[resp_status, respiratory_psd_plot]
            )

    return interface


# ============================================================================
# MAIN
# ============================================================================

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)