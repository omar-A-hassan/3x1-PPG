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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebSocket Receiver Service")

# Configuration
ESP32_DEVICE_NAME = "ESP32-PPG-Glucose"
PREPROCESSING_SERVICE_URL = "http://preprocessing:8001"
TOTAL_SAMPLES = 6000  # Total samples to collect

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

class CollectionRequest(BaseModel):
    action: str

class CollectionStatus(BaseModel):
    status: str
    samples_received: int
    total_samples: int
    last_saved_file: str | None = None

@app.on_event("startup")
async def _startup():
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    logger.info("Startup: captured event loop for thread callbacks")

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
        try:
            if len(data) < 4:
                logger.error("Received binary message too short")
                return

            chunk_num = struct.unpack('<H', data[0:2])[0]
            total_chunks = data[2]
            sample_count = data[3]

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

            # If collection complete, schedule processing
            if total_received >= TOTAL_SAMPLES and not save_triggered:
                save_triggered = True  # ✅ Prevent re-triggering
                collection_metadata["end_time"] = datetime.now()
                
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
    New workflow:
    1. Convert buffer to numpy array
    2. Unscale data
    3. SAVE data to disk (multiple formats)
    4. Send to preprocessing service
    """
    global ppg_buffer, collection_status

    try:
        logger.info(f"Starting save and process workflow for {len(ppg_buffer)} samples...")
        
        # Step 1: Convert to numpy array
        ppg_signal = np.array(ppg_buffer, dtype=np.float64)
        logger.info(f"Converted to numpy array: {ppg_signal.shape}")
        
        # Step 2: Unscale data (ESP32 divided by 4)
        ppg_signal = ppg_signal * 4
        logger.info(f"Unscaled data (×4): mean={ppg_signal.mean():.2f}, std={ppg_signal.std():.2f}")
        
        # Step 3: SAVE DATA TO DISK (NEW STEP)
        logger.info("=" * 60)
        logger.info("SAVING PPG DATA TO DISK")
        logger.info("=" * 60)
        
        save_result = await save_ppg_data(ppg_signal)
        
        if save_result["success"]:
            logger.info(f"✓ Data saved successfully!")
            logger.info(f"✓ Files created: {len(save_result['files'])}")
            for format_name, file_path in save_result['files'].items():
                logger.info(f"  - {format_name.upper()}: {file_path}")
        else:
            logger.error(f"✗ Failed to save data: {save_result.get('error')}")
            # Continue to preprocessing even if save fails
        
        logger.info("=" * 60)
        
        # Step 4: Send to preprocessing service
        logger.info("Sending data to preprocessing service...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{PREPROCESSING_SERVICE_URL}/preprocess",
                json={"signal": ppg_signal.tolist()}
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ Preprocessing complete: {result.get('num_segments')} segments")
                
                with state_lock:
                    collection_status = "COMPLETE"
                
                return {
                    "preprocessing": result,
                    "saved_data": save_result
                }
            else:
                logger.error(f"✗ Preprocessing failed: {response.text}")
                with state_lock:
                    collection_status = "ERROR"
                return {
                    "preprocessing": None,
                    "saved_data": save_result
                }

    except Exception as e:
        logger.exception(f"Error in save_and_process_data: {e}")
        with state_lock:
            collection_status = "ERROR"
        return None


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
# NEW ENDPOINT: List Saved Data Files
# ============================================================================

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


# ============================================================================
# NEW ENDPOINT: Download Specific File
# ============================================================================

from fastapi.responses import FileResponse

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)