"""
HTTP Receiver Service with Data Saving
Receives PPG data chunks from ESP32 via HTTP POST and saves/processes them.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import asyncio
import numpy as np
import struct
import logging
import threading
from datetime import datetime
from pathlib import Path
import json
import csv
from fastapi.responses import FileResponse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HTTP Receiver Service")

# ============================================================================
# DEBUG/DEPLOYMENT CONFIGURATION
# ============================================================================
# For cloud deployment: disable CSV saving (no persistent storage)
# For local debugging: set both to True
ENABLE_CSV_SAVE = False          # Disable for cloud (no disk persistence)
ENABLE_DOWNLOAD_ENDPOINTS = False  # Disable download endpoints for cloud
# ============================================================================

# Configuration
ESP32_DEVICE_NAME = "ESP32-PPG-Glucose"
PREPROCESSING_SERVICE_URL = os.getenv("PREPROCESSING_SERVICE_URL", "http://localhost:8001")
UI_SERVICE_URL = os.getenv("UI_SERVICE_URL", "http://localhost:8003")
TOTAL_SAMPLES = 3000  # 60 seconds × 50 Hz sampling

# Data storage configuration (only used if ENABLE_CSV_SAVE=True)
DATA_STORAGE_DIR = Path("ppg_data")
if ENABLE_CSV_SAVE:
    DATA_STORAGE_DIR.mkdir(exist_ok=True)

# Global state
save_triggered = False
ppg_buffer = []
collection_status = "ESP32 Not Connected"
main_event_loop: asyncio.AbstractEventLoop | None = None

# Collection history - stores each completed 3000-sample collection
# Each entry: {timestamp, samples, glucose, resp_rate, quality_score, csv_file, ...}
collection_history = []

# Track collection metadata
collection_metadata = {
    "start_time": None,
    "end_time": None,
    "sample_rate": 50,
    "duration_seconds": 60,
    "device_name": ESP32_DEVICE_NAME,
    "last_saved_file": None,
    # ESP32 precise timing (received via HTTP)
    "esp32_duration_seconds": None,
    "esp32_sample_rate": None,
}

state_lock = threading.Lock()


async def post_with_retry(client: httpx.AsyncClient, url: str, json=None, retries: int = 3, backoff: float = 1.0):
    """Helper: POST with retries and exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"POST attempt {attempt}/{retries} -> {url}")
            resp = await client.post(url, json=json)
            return resp
        except httpx.RequestError as re:
            logger.warning(f"POST attempt {attempt} failed: {re}")
            if attempt < retries:
                sleep_for = backoff * (2 ** (attempt - 1))
                logger.info(f"Retrying after {sleep_for:.1f}s...")
                await asyncio.sleep(sleep_for)
            else:
                logger.error(f"All {retries} POST attempts failed for {url}")
                raise
        except Exception as e:
            logger.exception(f"Unexpected error during POST attempt {attempt}: {e}")
            raise


def _response_to_dict(resp: object) -> dict:
    """Convert an httpx.Response with status 200 to a dict."""
    try:
        if isinstance(resp, httpx.Response) and getattr(resp, "status_code", None) == 200:
            text = getattr(resp, "text", "") or "{}"
            return json.loads(text)
    except Exception:
        pass
    return {"success": False}


class CollectionStatus(BaseModel):
    status: str
    samples_received: int
    total_samples: int
    last_saved_file: str | None = None


@app.on_event("startup")
async def _startup():
    """Startup: capture event loop."""
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    logger.info("[Startup] HTTP receiver service ready")
    logger.info("[Startup] Waiting for ESP32 to POST data chunks")


@app.get("/")
async def root():
    return {
        "service": "HTTP Receiver",
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


@app.post("/receive_chunk")
async def receive_chunk(request: Request):
    """
    Receive binary PPG data chunk via HTTP POST from ESP32.
    Automatically resets buffer when Chunk 0 arrives.
    Triggers processing when all samples are received.
    """
    global ppg_buffer, collection_status, save_triggered
    
    try:
        # Get metadata from headers
        chunk_num = int(request.headers.get("X-Chunk-Number", 0))
        total_chunks = int(request.headers.get("X-Total-Chunks", 1))
        
        # Read binary data
        data = await request.body()
        
        logger.info(f"[DEBUG] Received BINARY HTTP POST, length: {len(data)} bytes")
        
        if len(data) < 4:
            logger.error("Received binary message too short")
            raise HTTPException(status_code=400, detail="Invalid chunk data")
        
        # Parse header (same format as original)
        chunk_num_data = struct.unpack('<H', data[0:2])[0]
        total_chunks_data = data[2]
        sample_count_data = data[3]
        
        logger.info(f"[DEBUG] Chunk header: num={chunk_num_data}, total={total_chunks_data}, samples={sample_count_data}")
        
        # AUTOMATIC RESET: If this is the first chunk, clear buffer and reset state
        if chunk_num_data == 0:
            with state_lock:
                logger.info("Received Chunk 0 - Auto-resetting buffer for new collection")
                ppg_buffer = []
                collection_status = "COLLECTING"
                collection_metadata["start_time"] = datetime.now()
                collection_metadata["end_time"] = None
                collection_metadata["last_saved_file"] = None
                collection_metadata["esp32_duration_seconds"] = None
                collection_metadata["esp32_sample_rate"] = None
                save_triggered = False
        
        # Extract samples
        samples = []
        for i in range(sample_count_data):
            offset = 4 + i * 2
            if offset + 2 <= len(data):
                sample = struct.unpack('<H', data[offset:offset+2])[0]
                samples.append(sample)
        
        # Add to buffer
        with state_lock:
            ppg_buffer.extend(samples)
            total_received = len(ppg_buffer)
        
        logger.info(f"Received chunk {chunk_num + 1}/{total_chunks}: {len(samples)} samples (total: {total_received})")
        
        return {
            "success": True,
            "chunk": chunk_num,
            "samples_received": len(samples),
            "total_samples": total_received
        }
        
    except Exception as e:
        logger.exception(f"Error processing chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collection_complete")
async def collection_complete(request: Request):
    """
    Receive collection metadata from ESP32 after all chunks sent.
    This triggers the data processing pipeline.
    """
    global save_triggered
    
    try:
        data = await request.json()
        
        # Store ESP32's precise timing metadata
        collection_metadata["esp32_duration_seconds"] = data.get("duration_seconds")
        collection_metadata["esp32_sample_rate"] = data.get("actual_sample_rate")
        collection_metadata["end_time"] = datetime.now()
        
        duration = data.get('duration_seconds', 0)
        rate = data.get('actual_sample_rate', 0)
        logger.info(f"ESP32 timing: {duration:.2f}s @ {rate:.2f} Hz")
        
        # Trigger processing now that we have the metadata
        if not save_triggered:
            save_triggered = True
            logger.info("Collection complete - triggering save and processing...")
            if main_event_loop:
                asyncio.run_coroutine_threadsafe(
                    save_and_process_data(),
                    main_event_loop
                )
        
        return {"success": True}
    except Exception as e:
        logger.exception(f"Error processing collection_complete: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/status_update")
async def status_update(request: Request):
    """Receive status updates from ESP32 (optional, for monitoring)."""
    global collection_status
    
    try:
        data = await request.json()
        status_msg = data.get("status", "")
        
        with state_lock:
            collection_status = status_msg
        
        logger.info(f"ESP32 status update: {status_msg}")
        
        return {"success": True}
    except Exception as e:
        logger.exception(f"Error processing status: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# Save PPG Data to Multiple Formats
# ============================================================================

async def save_ppg_data(ppg_signal: np.ndarray) -> dict:
    """Save PPG data to disk in multiple formats."""
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

        # FORMAT 2: CSV Format
        csv_path = DATA_STORAGE_DIR / f"{base_filename}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample_Index', 'IR_Value', 'Timestamp_ms'])
            for i, value in enumerate(ppg_signal):
                timestamp_ms = (i / collection_metadata["sample_rate"]) * 1000
                writer.writerow([i, int(value), f"{timestamp_ms:.2f}"])
        saved_files["csv"] = str(csv_path)
        logger.info(f"Saved CSV: {csv_path}")
        
        # FORMAT 3: JSON Format - With metadata
        json_path = DATA_STORAGE_DIR / f"{base_filename}.json"
        json_data = {
            "metadata": metadata,
            "data": ppg_signal.tolist()
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        saved_files["json"] = str(json_path)
        logger.info(f"Saved JSON: {json_path}")
        
        # FORMAT 4: NPY Format - For fast numpy loading
        npy_path = DATA_STORAGE_DIR / f"{base_filename}.npy"
        np.save(npy_path, ppg_signal)
        saved_files["npy"] = str(npy_path)
        logger.info(f"Saved NPY: {npy_path}")
        
        # Update global metadata
        collection_metadata["last_saved_file"] = str(csv_path)
        
        logger.info(f"Successfully saved PPG data in {len(saved_files)} formats")
        
        return {
            "success": True,
            "timestamp": timestamp,
            "files": saved_files,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.exception(f"Error saving PPG data: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# Save Data THEN Process
# ============================================================================

async def save_and_process_data():
    """
    Streamlined workflow:
    1. Convert & unscale
    2. SAVE data to disk (if ENABLE_CSV_SAVE=True)
    3. Run /preprocess and /preprocess_respiratory in parallel
    4. Merge results and notify UI
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
        
        # Use ESP32's precise timing if available
        esp32_duration = collection_metadata.get("esp32_duration_seconds")
        esp32_rate = collection_metadata.get("esp32_sample_rate")
        
        if esp32_duration and esp32_rate:
            duration_seconds = float(esp32_duration)
            effective_fs = float(esp32_rate)
            logger.info(f"[Timing] Using ESP32 precise timing: duration={duration_seconds:.3f}s, fs={effective_fs:.2f} Hz")
        else:
            # Fallback: Compute from Python wall-clock timestamps
            start_time = collection_metadata.get("start_time")
            end_time = collection_metadata.get("end_time")
            duration_seconds = None
            if start_time and end_time:
                try:
                    duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
                except Exception:
                    duration_seconds = None
            if not duration_seconds or duration_seconds <= 0:
                duration_seconds = float(len(ppg_signal)) / 50.0 if len(ppg_signal) > 0 else 0.0
            effective_fs = float(len(ppg_signal)) / duration_seconds if duration_seconds > 0 else 50.0
            logger.warning(f"[Timing] ESP32 timing not available, using fallback: duration={duration_seconds:.3f}s, fs={effective_fs:.2f} Hz")

        # Update metadata for saving and downstream services
        collection_metadata["duration_seconds"] = float(duration_seconds)
        collection_metadata["sample_rate"] = float(effective_fs)

        csv_file_path = None
        
        # Step 3: OPTIONALLY SAVE DATA TO DISK
        if ENABLE_CSV_SAVE:
            logger.info("=" * 60)
            logger.info("SAVING PPG DATA TO DISK")
            logger.info("=" * 60)
            
            save_result = await save_ppg_data(ppg_signal)
            
            if save_result["success"]:
                logger.info("✓ Data saved successfully!")
                csv_file_path = save_result['files'].get('csv')
                logger.info(f"✓ CSV saved at: {csv_file_path}")
                for format_name, file_path in save_result['files'].items():
                    logger.info(f"  - {format_name.upper()}: {file_path}")
            else:
                logger.error(f"✗ Failed to save data: {save_result.get('error')}")
            
            logger.info("=" * 60)
        else:
            logger.info("CSV saving disabled (ENABLE_CSV_SAVE=False)")
        
        # Step 4: Run glucose and respiratory preprocessing in parallel
        logger.info("Sending data to preprocessing service...")

        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                glucose_task = post_with_retry(
                    client,
                    f"{PREPROCESSING_SERVICE_URL}/preprocess",
                    json={"signal": ppg_signal.tolist()},
                    retries=3,
                    backoff=1.0,
                )
                resp_task = post_with_retry(
                    client,
                    f"{PREPROCESSING_SERVICE_URL}/preprocess_respiratory",
                    json={
                        "signal": ppg_signal.tolist(),
                        "sampling_rate": int(round(effective_fs)),
                    },
                    retries=3,
                    backoff=1.0,
                )

                glucose_response, resp_response = await asyncio.gather(glucose_task, resp_task, return_exceptions=True)

                # Handle glucose response
                glucose_json = {"success": False}
                if isinstance(glucose_response, Exception):
                    logger.error(f"Glucose preprocessing exception: {glucose_response}")
                elif isinstance(glucose_response, httpx.Response):
                    status = getattr(glucose_response, "status_code", None)
                    if status == 200:
                        glucose_json = _response_to_dict(glucose_response)
                    else:
                        txt = getattr(glucose_response, "text", "")
                        logger.error(f"Glucose preprocessing returned {status}: {txt}")

                # Handle respiratory response
                resp_json = {"success": False}
                if isinstance(resp_response, Exception):
                    logger.error(f"Respiratory preprocessing exception: {resp_response}")
                elif isinstance(resp_response, httpx.Response):
                    status = getattr(resp_response, "status_code", None)
                    if status == 200:
                        resp_json = _response_to_dict(resp_response)
                    else:
                        txt = getattr(resp_response, "text", "")
                        logger.error(f"Respiratory preprocessing returned {status}: {txt}")

                # Build unified result for UI
                dt_ms = 1000.0 / float(int(round(effective_fs)) if effective_fs else 100.0)
                timestamps_ms = [i * dt_ms for i in range(len(ppg_signal))]
                glucose_value = glucose_json.get("glucose") if isinstance(glucose_json, dict) else None
                model_device = glucose_json.get("model_device") if isinstance(glucose_json, dict) else None

                combined = {
                    "success": True,
                    # Glucose
                    "glucose": glucose_value,
                    "num_segments": int(glucose_json.get("num_segments", 0) or 0),
                    "quality_score": float(glucose_json.get("quality_score", 0.0) or 0.0),
                    # Respiratory
                    "resp_rate_bpm": resp_json.get("resp_rate_bpm"),
                    "resp_freq_hz": resp_json.get("resp_freq_hz"),
                    "esqi": resp_json.get("esqi"),
                    "entropy": resp_json.get("entropy"),
                    "peaks_count": int(resp_json.get("peaks_count", 0) or 0),
                    "method_used": resp_json.get("method_used"),
                    "freqs": resp_json.get("freqs"),
                    "psd": resp_json.get("psd"),
                    # Raw signal and timestamps for UI plotting
                    "raw_signal": ppg_signal.tolist(),
                    "timestamps_ms": timestamps_ms,
                    # Metadata
                    "device": model_device or "receiver",
                    "csv_file": csv_file_path,
                }

                # Notify UI
                try:
                    ui_resp = await post_with_retry(
                        client,
                        f"{UI_SERVICE_URL}/update_result",
                        json=combined,
                        retries=2,
                        backoff=0.5,
                    )
                    if isinstance(ui_resp, httpx.Response):
                        if getattr(ui_resp, "status_code", None) == 200:
                            logger.info("✓ UI updated with combined result")
                        else:
                            logger.warning("UI update did not return 200")
                    else:
                        logger.warning("UI update call did not return a Response object")
                except Exception as ui_e:
                    logger.warning(f"Failed to update UI: {ui_e}")

                logger.info("✓ Parallel preprocessing complete")
                logger.info(f"  - Segments: {combined['num_segments']}")
                logger.info(f"  - Respiratory Rate: {combined.get('resp_rate_bpm')} bpm")

                # Append to collection history
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "samples": len(ppg_signal),
                    "glucose": glucose_value,
                    "resp_rate_bpm": resp_json.get("resp_rate_bpm"),
                    "quality_score": float(glucose_json.get("quality_score", 0.0) or 0.0),
                    "num_segments": int(glucose_json.get("num_segments", 0) or 0),
                    "csv_file": csv_file_path,
                    "esp32_duration": esp32_duration,
                    "esp32_rate": esp32_rate,
                }
                with state_lock:
                    collection_history.append(history_entry)
                    collection_status = "COMPLETE"
                logger.info(f"✓ Added to collection history (total: {len(collection_history)} entries)")

                return {
                    "success": True,
                    "preprocessing_result": combined,
                    "csv_file": csv_file_path,
                }

            except Exception as e:
                logger.error(f"Preprocessing requests failed: {e}")
                with state_lock:
                    collection_status = "ERROR"
                return {"success": False, "error": str(e)}

    except Exception as e:
        logger.exception(f"Error in save_and_process_data: {e}")
        with state_lock:
            collection_status = "ERROR"
        return {"success": False, "error": str(e)}


# ============================================================================
# COLLECTION HISTORY ENDPOINT
# ============================================================================

@app.get("/collection_history")
async def get_collection_history():
    """Get history of all completed collections."""
    with state_lock:
        history_copy = list(collection_history)
    return {
        "total_collections": len(history_copy),
        "history": history_copy
    }


# ============================================================================
# DOWNLOAD ENDPOINTS (conditionally registered)
# ============================================================================

if ENABLE_DOWNLOAD_ENDPOINTS:
    @app.get("/saved_data")
    async def list_saved_data():
        """List all saved PPG data files."""
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
        """Download a specific saved data file."""
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
