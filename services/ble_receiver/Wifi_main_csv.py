"""
WebSocket Receiver Service with Data Saving
Saves collected PPG data before sending to preprocessing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
import numpy as np
import struct
import logging
import threading
import time
from websocket import WebSocketApp
from datetime import datetime
from pathlib import Path
import json
import csv
from fastapi.responses import FileResponse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebSocket Receiver Service")

# ============================================================================
# DEBUG/DEPLOYMENT CONFIGURATION
# ============================================================================
ENABLE_CSV_SAVE = False          # Save all 5 formats to disk (archival)
ENABLE_DOWNLOAD_ENDPOINTS = False  # Expose /download and /saved_data endpoints
# ============================================================================

# Configuration
ESP32_DEVICE_NAME = "ESP32-PPG-Glucose"
PREPROCESSING_SERVICE_URL = os.getenv("PREPROCESSING_SERVICE_URL", "http://localhost:8001")
UI_SERVICE_URL = os.getenv("UI_SERVICE_URL", "http://localhost:8003")
TOTAL_SAMPLES = 9000  # 180 seconds × 50 Hz (optimal for smooth PPG + accurate respiratory)

# Data storage configuration
DATA_STORAGE_DIR = Path("ppg_data")  # Directory to store collected data
DATA_STORAGE_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist

# Global state
connected_ws_app: WebSocketApp | None = None
ws_thread: threading.Thread | None = None
ws_url_in_use: str | None = None
save_triggered = False

ppg_buffer = []
collection_status = "IDLE"
main_event_loop: asyncio.AbstractEventLoop | None = None

# Track collection metadata
collection_metadata = {
    "start_time": None,
    "end_time": None,
    "sample_rate": 100,
    "duration_seconds": 60,
    "device_name": ESP32_DEVICE_NAME,
    "last_saved_file": None
}

state_lock = threading.Lock()


async def post_with_retry(client: httpx.AsyncClient, url: str, json=None, retries: int = 3, backoff: float = 1.0):
    """Helper: POST with retries and exponential backoff.

    Raises the last exception if all retries fail.
    Returns the httpx.Response on success.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"POST attempt {attempt}/{retries} -> {url}")
            resp = await client.post(url, json=json)
            return resp
        except httpx.RequestError as re:
            last_exc = re
            logger.warning(f"POST attempt {attempt} failed: {re}")
            if attempt < retries:
                sleep_for = backoff * (2 ** (attempt - 1))
                logger.info(f"Retrying after {sleep_for:.1f}s...")
                await asyncio.sleep(sleep_for)
            else:
                logger.error(f"All {retries} POST attempts failed for {url}")
                raise
        except Exception as e:
            last_exc = e
            logger.exception(f"Unexpected error during POST attempt {attempt}: {e}")
            raise


class CollectionRequest(BaseModel):
    action: str

class CollectionStatus(BaseModel):
    status: str
    samples_received: int
    total_samples: int
    last_saved_file: str | None = None

@app.on_event("startup")
async def _startup():
    """
    Startup: capture event loop and try automatic ESP32 connection.
    """
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    logger.info("[Startup] Captured event loop for thread callbacks")

    ws_url = "ws://192.168.4.1:81/"
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[Auto-Connect] Attempt {attempt} to connect to ESP32 at {ws_url}")
            _start_ws_thread(ws_url)
            await asyncio.sleep(2.0)  # give connection time
            logger.info(f"[Auto-Connect] Thread started (attempt {attempt})")
            break
        except Exception as e:
            logger.warning(f"[Auto-Connect] Attempt {attempt} failed: {e}")
            await asyncio.sleep(3)
    else:
        logger.error(f"[Auto-Connect] All {max_retries} attempts failed.")

@app.get("/")
async def root():
    return {
        "service": "WebSocket Receiver",
        "status": collection_status,
        "samples_received": len(ppg_buffer),
        "last_saved_file": collection_metadata.get("last_saved_file")
    }

@app.get("/status")
async def get_status():
    return CollectionStatus(
        status=collection_status,
        samples_received=len(ppg_buffer),
        total_samples=TOTAL_SAMPLES,
        last_saved_file=collection_metadata.get("last_saved_file")
    )

@app.post("/scan")
async def scan_devices():
    example_ws = "ws://192.168.4.1:81/"
    return {
        "note": "In AP/WebSocket mode, scan is not available.",
        "example_ws_url": example_ws,
        "ssid_hint": "ESP32-PPG-Glucose (AP mode)"
    }

def _on_ws_open(ws):
    global collection_status
    logger.info("WebSocket connection opened")
    with state_lock:
        collection_status = "CONNECTED"

def _on_ws_close(ws, close_status_code, close_msg):
    global collection_status
    logger.info(f"WebSocket closed: code={close_status_code}, msg={close_msg}")
    with state_lock:
        collection_status = "DISCONNECTED"

def _on_ws_error(ws, error):
    global collection_status
    logger.error(f"WebSocket error: {error}")
    with state_lock:
        collection_status = "ERROR"

def _on_ws_message(ws, message):
    global ppg_buffer, collection_status

    # Binary data (chunks)
    if isinstance(message, (bytes, bytearray)):
        data = message
        logger.info(f"[DEBUG] Received BINARY message, length: {len(data)} bytes")
        try:
            if len(data) < 4:
                logger.error("Received binary message too short")
                return

            chunk_num = struct.unpack('<H', data[0:2])[0]
            total_chunks = data[2]
            sample_count = data[3]
            
            logger.info(f"[DEBUG] Chunk header: num={chunk_num}, total={total_chunks}, samples={sample_count}")

            samples = []
            for i in range(sample_count):
                offset = 4 + i * 2
                if offset + 2 <= len(data):
                    sample = struct.unpack('<H', data[offset:offset+2])[0]
                    samples.append(sample)

            with state_lock:
                ppg_buffer.extend(samples)
                total_received = len(ppg_buffer)

            logger.info(f"Received chunk {chunk_num + 1}/{total_chunks}: {len(samples)} samples (total: {total_received})")

            global save_triggered

            # If collection complete, schedule processing and stop device streaming
            if total_received >= TOTAL_SAMPLES and not save_triggered:
                save_triggered = True  # ✅ Prevent re-triggering
                collection_metadata["end_time"] = datetime.now()

                # Proactively send STOP/CANCEL to ESP32 to halt further streaming
                try:
                    ws.send("C")
                    logger.info("Sent STOP command to ESP32 (TOTAL_SAMPLES reached)")
                except Exception as e:
                    logger.warning(f"Failed to send STOP to ESP32: {e}")

                logger.info("Buffer reached TOTAL_SAMPLES, scheduling save and processing")
                if main_event_loop:
                    asyncio.run_coroutine_threadsafe(
                        save_and_process_data(),
                        main_event_loop
                    )

        except Exception as e:
            logger.exception(f"Error processing binary message: {e}")
        return

    # Text messages
    try:
        msg = message
        logger.debug(f"Text WS message: {msg}")

        if isinstance(msg, str) and msg.strip().startswith('{'):
            import json
            try:
                js = json.loads(msg)
                ev = js.get("event")
                if ev == "status":
                    status_val = js.get("status")
                    if status_val:
                        with state_lock:
                            collection_status = status_val
                        logger.info(f"ESP32 status: {status_val}")
                elif ev == "progress":
                    logger.info(f"ESP32 progress: {js.get('progress')}%")
                elif ev == "transmission":
                    logger.info(f"Transmission progress: {js.get('progress')}")
            except Exception as je:
                logger.debug(f"Failed to parse JSON: {je}")
                with state_lock:
                    collection_status = msg
        else:
            with state_lock:
                collection_status = msg
            logger.info(f"ESP32 Status: {msg}")

    except Exception as e:
        logger.exception(f"Error processing text message: {e}")

def _start_ws_thread(ws_url):
    global connected_ws_app, ws_thread, ws_url_in_use

    if connected_ws_app is not None:
        logger.info("WebSocket client already running")
        return

    logger.info(f"Starting WebSocket client for {ws_url}")
    connected_ws_app = WebSocketApp(
        ws_url,
        on_open=_on_ws_open,
        on_message=_on_ws_message,
        on_error=_on_ws_error,
        on_close=_on_ws_close
    )

    def run_ws():
        try:
            connected_ws_app.run_forever()
        except Exception as e:
            logger.exception(f"WebSocket run_forever terminated: {e}")
        finally:
            logger.info("WebSocket thread terminated")
            with state_lock:
                global collection_status
                collection_status = "DISCONNECTED"

    ws_thread = threading.Thread(target=run_ws, name="ws-thread", daemon=True)
    ws_thread.start()
    ws_url_in_use = ws_url

@app.post("/connect")
async def connect_device(device_address: str):
    global connected_ws_app, ws_thread, ws_url_in_use, collection_status

    if not device_address:
        raise HTTPException(status_code=400, detail="device_address required")

    if connected_ws_app is not None:
        return {"status": "already_connected", "ws_url": ws_url_in_use}

    try:
        _start_ws_thread(device_address)
        await asyncio.sleep(0.5)
        return {"status": "connecting", "ws_url": device_address}
    except Exception as e:
        logger.exception(f"Failed to start WebSocket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect")
async def start_collection():
    global ppg_buffer, collection_status, connected_ws_app, collection_metadata

    with state_lock:
        if connected_ws_app is None:
            raise HTTPException(status_code=400, detail="Not connected")

    try:
        with state_lock:
            ppg_buffer = []
            collection_status = "COLLECTING"
            # Record start time
            collection_metadata["start_time"] = datetime.now()
            collection_metadata["end_time"] = None
            collection_metadata["last_saved_file"] = None
            # Reset flags and set defaults; will be corrected after capture
            global save_triggered
            save_triggered = False
            collection_metadata["sample_rate"] = 100
            collection_metadata["duration_seconds"] = 60

        if connected_ws_app:
            try:
                connected_ws_app.send("S")
                logger.info("Sent START command to ESP32")
            except Exception as e:
                logger.exception(f"Failed to send start command: {e}")
                raise

        return {"status": "collecting"}
    except Exception as e:
        logger.exception(f"Failed to start collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NEW FUNCTION: Save PPG Data to Multiple Formats
# ============================================================================

async def save_ppg_data(ppg_signal: np.ndarray) -> dict:
    """
    Save PPG data to disk in multiple formats
    
    Args:
        ppg_signal: Unscaled PPG data as numpy array
    
    Returns:
        dict with file paths and metadata
    """
    try:
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"ppg_data_{timestamp}"
        
        # Prepare metadata
        metadata = {
            "timestamp": timestamp,
            "start_time": collection_metadata["start_time"].isoformat() if collection_metadata["start_time"] else None,
            "end_time": collection_metadata["end_time"].isoformat() if collection_metadata["end_time"] else None,
            "sample_rate": collection_metadata["sample_rate"],
            "total_samples": len(ppg_signal),
            "duration_seconds": collection_metadata["duration_seconds"],
            "device_name": collection_metadata["device_name"],
            "signal_stats": {
                "mean": float(ppg_signal.mean()),
                "std": float(ppg_signal.std()),
                "min": float(ppg_signal.min()),
                "max": float(ppg_signal.max()),
                "median": float(np.median(ppg_signal))
            }
        }
        
        saved_files = {}
        
        # ======================================================================
        # FORMAT 1: NumPy Binary Format (.npy) - Fast loading, preserves dtype
        # ======================================================================
        npy_path = DATA_STORAGE_DIR / f"{base_filename}.npy"
        np.save(npy_path, ppg_signal)
        saved_files["npy"] = str(npy_path)
        logger.info(f"Saved NumPy binary: {npy_path}")
        
        # ======================================================================
        # FORMAT 2: CSV Format - Human readable, universal compatibility
        # ======================================================================
        csv_path = DATA_STORAGE_DIR / f"{base_filename}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Sample_Index', 'IR_Value', 'Timestamp_ms'])
            # Write data
            for i, value in enumerate(ppg_signal):
                timestamp_ms = (i / collection_metadata["sample_rate"]) * 1000
                writer.writerow([i, int(value), f"{timestamp_ms:.2f}"])
        saved_files["csv"] = str(csv_path)
        logger.info(f"Saved CSV: {csv_path}")
        
        # ======================================================================
        # FORMAT 3: JSON Format - With metadata, easily parseable
        # ======================================================================
        json_path = DATA_STORAGE_DIR / f"{base_filename}.json"
        json_data = {
            "metadata": metadata,
            "data": ppg_signal.tolist()  # Convert numpy array to Python list
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        saved_files["json"] = str(json_path)
        logger.info(f"Saved JSON: {json_path}")
        
        # ======================================================================
        # FORMAT 4: Metadata-only JSON - Quick reference
        # ======================================================================
        meta_path = DATA_STORAGE_DIR / f"{base_filename}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files["metadata"] = str(meta_path)
        logger.info(f"Saved metadata: {meta_path}")
        
        # ======================================================================
        # FORMAT 5: Text Summary - Human-readable report
        # ======================================================================
        summary_path = DATA_STORAGE_DIR / f"{base_filename}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PPG DATA COLLECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Collection Timestamp: {timestamp}\n")
            f.write(f"Start Time: {metadata['start_time']}\n")
            f.write(f"End Time: {metadata['end_time']}\n")
            f.write(f"Device: {metadata['device_name']}\n\n")
            f.write(f"Sample Rate: {metadata['sample_rate']} Hz\n")
            f.write(f"Total Samples: {metadata['total_samples']}\n")
            f.write(f"Duration: {metadata['duration_seconds']} seconds\n\n")
            f.write("Signal Statistics:\n")
            f.write(f"  Mean:   {metadata['signal_stats']['mean']:.2f}\n")
            f.write(f"  Std:    {metadata['signal_stats']['std']:.2f}\n")
            f.write(f"  Min:    {metadata['signal_stats']['min']:.2f}\n")
            f.write(f"  Max:    {metadata['signal_stats']['max']:.2f}\n")
            f.write(f"  Median: {metadata['signal_stats']['median']:.2f}\n\n")
            f.write("=" * 60 + "\n")
            f.write("Data Files:\n")
            f.write("=" * 60 + "\n")
            for format_name, file_path in saved_files.items():
                f.write(f"  {format_name.upper()}: {file_path}\n")
        saved_files["summary"] = str(summary_path)
        logger.info(f"Saved summary: {summary_path}")
        
        # Update global metadata
        collection_metadata["last_saved_file"] = str(npy_path)
        
        logger.info(f"Successfully saved PPG data in {len(saved_files)} formats")
        
        return {
            "success": True,
            "timestamp": timestamp,
            "files": saved_files,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.exception(f"Error saving PPG data: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# MODIFIED FUNCTION: Save Data THEN Process
# ============================================================================

async def save_and_process_data():
    """
    Streamlined workflow:
    1. Convert & unscale
    2. SAVE data to disk (if ENABLE_CSV_SAVE=True)
    3. Send to preprocessing /preprocess_combined
    4. Preprocessing handles model inference, respiratory analysis, and UI update
    """
    global ppg_buffer, collection_status

    try:
        logger.info(f"Starting save and process workflow for {len(ppg_buffer)} samples...")
        
        # Step 1: Convert to numpy array and hard-cap to TOTAL_SAMPLES
        with state_lock:
            buf_copy = list(ppg_buffer)
        if len(buf_copy) >= TOTAL_SAMPLES:
            buf_copy = buf_copy[:TOTAL_SAMPLES]
        ppg_signal = np.array(buf_copy, dtype=np.float64)
        logger.info(f"Converted to numpy array (capped): {ppg_signal.shape}")
        
        # Step 2: Unscale data (ESP32 divided by 4)
        ppg_signal = ppg_signal * 4
        logger.info(f"Unscaled data: mean={ppg_signal.mean():.2f}, std={ppg_signal.std():.2f}")
        
        # Compute effective duration and sampling rate from start/end times
        start_time = collection_metadata.get("start_time")
        end_time = collection_metadata.get("end_time")
        duration_seconds = None
        if start_time and end_time:
            try:
                duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
            except Exception:
                duration_seconds = None
        if not duration_seconds or duration_seconds <= 0:
            # Fallback: assume near 100 Hz
            duration_seconds = float(len(ppg_signal)) / 100.0 if len(ppg_signal) > 0 else 0.0
        effective_fs = float(len(ppg_signal)) / duration_seconds if duration_seconds > 0 else 100.0

        logger.info(
            f"[Timing] start={start_time}, end={end_time}, duration={duration_seconds:.3f}s, fs_eff={effective_fs:.2f} Hz"
        )

        # Update metadata for saving and downstream services
        collection_metadata["duration_seconds"] = float(duration_seconds)
        collection_metadata["sample_rate"] = float(effective_fs)

        csv_file_path = None
        
        # Step 3: OPTIONALLY SAVE DATA TO DISK (archival)
        if ENABLE_CSV_SAVE:
            logger.info("=" * 60)
            logger.info("SAVING PPG DATA TO DISK")
            logger.info("=" * 60)
            
            save_result = await save_ppg_data(ppg_signal)
            
            if save_result["success"]:
                logger.info(f"✓ Data saved successfully!")
                csv_file_path = save_result['files'].get('csv')
                logger.info(f"✓ CSV saved at: {csv_file_path}")
                for format_name, file_path in save_result['files'].items():
                    logger.info(f"  - {format_name.upper()}: {file_path}")
            else:
                logger.error(f"✗ Failed to save data: {save_result.get('error')}")
            
            logger.info("=" * 60)
        else:
            logger.info("CSV saving disabled (ENABLE_CSV_SAVE=False)")
        
    # Step 4: Send to preprocessing /preprocess_combined (pass effective sampling_rate)
        logger.info("Sending data to preprocessing service /preprocess_combined...")
        logger.info("Preprocessing will run glucose+respiratory in parallel and notify UI")
        
        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                preprocessing_response = await post_with_retry(
                    client,
                    f"{PREPROCESSING_SERVICE_URL}/preprocess_combined",
                    json={
                        "signal": ppg_signal.tolist(),
                        "sampling_rate": int(round(effective_fs)),
                        "csv_file": csv_file_path
                    },
                    retries=3,
                    backoff=1.0,
                )
                
                if preprocessing_response.status_code == 200:
                    result = preprocessing_response.json()
                    logger.info(f"✓ Combined preprocessing complete!")
                    logger.info(f"  - Glucose: {result.get('glucose')} mg/dL")
                    logger.info(f"  - Segments: {result.get('num_segments')}")
                    logger.info(f"  - Respiratory Rate: {result.get('resp_rate_bpm')} bpm")
                    logger.info(f"  - UI notified by preprocessing service")
                    
                    with state_lock:
                        collection_status = "COMPLETE"
                    
                    return {
                        "success": True,
                        "preprocessing_result": result,
                        "csv_file": csv_file_path
                    }
                else:
                    logger.error(f"Preprocessing returned status {preprocessing_response.status_code}: {preprocessing_response.text}")
                    with state_lock:
                        collection_status = "COMPLETE_WITH_WARNINGS"
                    return {"success": False, "error": f"Status {preprocessing_response.status_code}"}
                    
            except Exception as e:
                logger.error(f"Preprocessing request failed after retries: {e}")
                with state_lock:
                    collection_status = "ERROR"
                return {"success": False, "error": str(e)}

    except Exception as e:
        logger.exception(f"Error in save_and_process_data: {e}")
        with state_lock:
            collection_status = "ERROR"
        return {"success": False, "error": str(e)}


@app.post("/stop")
async def stop_collection():
    global connected_ws_app, collection_status

    with state_lock:
        if connected_ws_app is None:
            raise HTTPException(status_code=400, detail="Not connected")

    try:
        if connected_ws_app:
            connected_ws_app.send("C")
            logger.info("Sent STOP command to ESP32")
        with state_lock:
            collection_status = "STOPPED"
        return {"status": "stopped"}
    except Exception as e:
        logger.exception(f"Failed to stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disconnect")
async def disconnect_device():
    global connected_ws_app, ws_thread, ws_url_in_use, collection_status

    if connected_ws_app:
        try:
            connected_ws_app.close()
            time.sleep(0.2)
        except Exception as e:
            logger.exception(f"Error closing WebSocket: {e}")

    connected_ws_app = None
    ws_thread = None
    ws_url_in_use = None
    with state_lock:
        collection_status = "DISCONNECTED"
    logger.info("Disconnected from ESP32")
    return {"status": "disconnected"}


# ============================================================================
# DOWNLOAD ENDPOINTS (conditionally registered based on ENABLE_DOWNLOAD_ENDPOINTS)
# ============================================================================

if ENABLE_DOWNLOAD_ENDPOINTS:
    @app.get("/saved_data")
    async def list_saved_data():
        """
        List all saved PPG data files
        """
        try:
            files = {
                "npy": sorted(DATA_STORAGE_DIR.glob("*.npy")),
                "csv": sorted(DATA_STORAGE_DIR.glob("*.csv")),
                "json": sorted(DATA_STORAGE_DIR.glob("*.json")),
                "summary": sorted(DATA_STORAGE_DIR.glob("*_summary.txt"))
            }
            
            file_list = {}
            for format_name, paths in files.items():
                file_list[format_name] = [str(p) for p in paths]
            
            return {
                "storage_directory": str(DATA_STORAGE_DIR),
                "total_files": sum(len(v) for v in file_list.values()),
                "files_by_format": file_list
            }
        except Exception as e:
            logger.exception(f"Error listing saved data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/download/{filename}")
    async def download_file(filename: str):
        """
        Download a specific saved data file
        """
        file_path = DATA_STORAGE_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
else:
    logger.info("Download endpoints disabled (ENABLE_DOWNLOAD_ENDPOINTS=False)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)