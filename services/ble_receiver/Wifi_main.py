"""
WebSocket Receiver Service (converted from BLE Receiver)
Receives PPG data from ESP32 via WebSocket and forwards to preprocessing service
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
from websocket import WebSocketApp  # websocket-client library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebSocket Receiver Service")


# Configuration
# NOTE: device_address (in /connect) should now be the websocket URL,
# e.g. "ws://192.168.4.1:81/" (kept param name to preserve API)
ESP32_DEVICE_NAME = "ESP32-PPG-Glucose"  # kept for compatibility/clarity
PREPROCESSING_SERVICE_URL = "http://preprocessing:8001"
TOTAL_SAMPLES = 6000

# Global state
connected_ws_app: WebSocketApp | None = None
ws_thread: threading.Thread | None = None
ws_url_in_use: str | None = None

ppg_buffer = []
collection_status = "IDLE"
main_event_loop: asyncio.AbstractEventLoop | None = None

# Lock for buffer & state safety
state_lock = threading.Lock()

class CollectionRequest(BaseModel):
    action: str  # "start" or "stop"

class CollectionStatus(BaseModel):
    status: str
    samples_received: int
    total_samples: int

@app.on_event("startup")
async def _startup():
    """
    Startup: capture event loop and try automatic ESP32 connection.
    """
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    logger.info("Startup: captured event loop for thread callbacks")

    ws_url = "ws://192.168.4.1:81/"
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[Auto-Connect] Attempt {attempt} to connect to ESP32 at {ws_url}")
            _start_ws_thread(ws_url)
            await asyncio.sleep(2.0)  # allow connection time
            logger.info("[Auto-Connect] Thread started — waiting for connection...")
            await asyncio.sleep(2.0)
            logger.info("[Auto-Connect] Success — ESP32 should now be connected.")
            break
        except Exception as e:
            logger.warning(f"[Auto-Connect] Attempt {attempt} failed: {e}")
            await asyncio.sleep(3.0)
    else:
        logger.error(f"[Auto-Connect] Failed to connect after {max_retries} attempts.")


@app.get("/")
async def root():
    return {
        "service": "WebSocket Receiver",
        "status": collection_status,
        "samples_received": len(ppg_buffer)
    }

@app.get("/status")
async def get_status():
    return CollectionStatus(
        status=collection_status,
        samples_received=len(ppg_buffer),
        total_samples=TOTAL_SAMPLES
    )

@app.post("/scan")
async def scan_devices():
    """
    BLE scan is not applicable for WebSocket/AP mode.
    Return the example connection info for the ESP32 AP + WebSocket.
    """
    example_ws = "ws://192.168.4.1:81/"
    return {
        "note": "In AP/WebSocket mode, scan is not available. Use /connect with the WebSocket URL.",
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
    """
    Called for text messages (str) or binary (bytes) messages.
    websocket-client will pass bytes for binary frames.
    """
    global ppg_buffer, collection_status

    # Binary data (chunks)
    if isinstance(message, (bytes, bytearray)):
        data = message
        try:
            # Parse chunk header: [chunk_num (2 bytes LE), total_chunks (1), sample_count (1), ...data...]
            if len(data) < 4:
                logger.error("Received binary message too short to contain header")
                return

            chunk_num = struct.unpack('<H', data[0:2])[0]
            total_chunks = data[2]
            sample_count = data[3]

            # Safety check
            expected_len = 4 + sample_count * 2
            if len(data) < expected_len:
                logger.warning(f"Binary chunk length {len(data)} smaller than expected {expected_len}")

            samples = []
            for i in range(sample_count):
                offset = 4 + i * 2
                if offset + 2 <= len(data):
                    sample = struct.unpack('<H', data[offset:offset+2])[0]
                    samples.append(sample)
                else:
                    break

            with state_lock:
                ppg_buffer.extend(samples)
                total_received = len(ppg_buffer)

            logger.info(f"Received chunk {chunk_num + 1}/{total_chunks}: {len(samples)} samples (total: {total_received})")

            # If collection complete, schedule processing on main event loop
            if total_received >= TOTAL_SAMPLES:
                logger.info("Buffer reached TOTAL_SAMPLES, scheduling processing task")
                if main_event_loop:
                    asyncio.run_coroutine_threadsafe(process_collected_data(), main_event_loop)
                else:
                    # fallback (should not happen)
                    asyncio.get_event_loop().create_task(process_collected_data())

        except Exception as e:
            logger.exception(f"Error processing binary message: {e}")
        return

    # Text message(s)
    try:
        # Attempt to decode as JSON-like status (ESP32 sends JSON strings for status/progress)
        # message here is str
        msg = message
        logger.debug(f"Text WS message: {msg}")

        # Try parse lightweight: if it starts with '{', treat as JSON
        if isinstance(msg, str) and msg.strip().startswith('{'):
            import json
            try:
                js = json.loads(msg)
                # Example JSON formats your ESP32 sends:
                # {"event":"status","status":"READY",...}
                # {"event":"progress", "progress": ...}
                ev = js.get("event")
                if ev == "status":
                    status_val = js.get("status")
                    if status_val:
                        with state_lock:
                            collection_status = status_val
                        logger.info(f"ESP32 status (json): {status_val}")
                elif ev == "progress":
                    # You can log or ignore progress notifications
                    logger.info(f"ESP32 progress: {js.get('progress')}%")
                elif ev == "transmission":
                    logger.info(f"Transmission progress: {js.get('progress')}")
                else:
                    logger.info(f"Received JSON event: {js}")
            except Exception as je:
                logger.debug(f"Failed to parse JSON text message: {je}; raw: {msg}")
                # fallback: treat as raw status text
                with state_lock:
                    collection_status = msg
                logger.info(f"ESP32 Status (raw): {msg}")
        else:
            # Non-JSON status string
            with state_lock:
                collection_status = msg
            logger.info(f"ESP32 Status (raw): {msg}")

    except Exception as e:
        logger.exception(f"Error processing text message: {e}")

def _start_ws_thread(ws_url):
    """
    Create WebSocketApp and run it in a background thread.
    """
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
        # run_forever blocks; it will call callbacks for messages
        try:
            connected_ws_app.run_forever()
        except Exception as e:
            logger.exception(f"WebSocket run_forever terminated with exception: {e}")
        finally:
            logger.info("WebSocket thread terminated")
            # ensure state noted
            with state_lock:
                global collection_status
                collection_status = "DISCONNECTED"

    ws_thread = threading.Thread(target=run_ws, name="ws-thread", daemon=True)
    ws_thread.start()
    ws_url_in_use = ws_url

@app.post("/connect")
async def connect_device(device_address: str):
    """
    Connect to ESP32 WebSocket server.
    device_address should be the ws URL, e.g. "ws://192.168.4.1:81/".
    (We kept the parameter name 'device_address' for API compatibility.)
    """
    global connected_ws_app, ws_thread, ws_url_in_use, collection_status

    if not device_address:
        raise HTTPException(status_code=400, detail="device_address must be a WebSocket URL, e.g. ws://192.168.4.1:81/")

    # If already running, return info
    if connected_ws_app is not None:
        return {"status": "already_connected", "ws_url": ws_url_in_use}

    try:
        _start_ws_thread(device_address)
        # Wait a short time for connection to establish (non-blocking behavior kept small)
        await asyncio.sleep(0.5)
        return {"status": "connecting", "ws_url": device_address}
    except Exception as e:
        logger.exception(f"Failed to start WebSocket client: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect")
async def start_collection():
    """
    Start PPG data collection: reset buffer and send 'S' command to ESP32 via WebSocket.
    """
    global ppg_buffer, collection_status, connected_ws_app

    with state_lock:
        if connected_ws_app is None:
            raise HTTPException(status_code=400, detail="Not connected to ESP32 WebSocket")

    try:
        # Reset buffer and state
        with state_lock:
            ppg_buffer = []
            collection_status = "COLLECTING"

        # Send START command 'S' (ESP32 expects single-char commands)
        # Sending as text frame
        if connected_ws_app:
            try:
                connected_ws_app.send("S")
                logger.info("Sent START command ('S') to ESP32")
            except Exception as e:
                logger.exception(f"Failed to send start command over WS: {e}")
                raise

        return {"status": "collecting"}
    except Exception as e:
        logger.exception(f"Failed to start collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_collected_data():
    """Send collected data to preprocessing service (same logic as original)"""
    global ppg_buffer, collection_status

    try:
        logger.info(f"Processing {len(ppg_buffer)} samples...")

        # Convert to numpy array
        ppg_signal = np.array(ppg_buffer, dtype=np.float64)

        # Scale back (ESP32 divided by 4 to fit 16-bit)
        ppg_signal = ppg_signal * 4

        logger.info(f"Signal stats: mean={ppg_signal.mean():.2f}, std={ppg_signal.std():.2f}")

        # Send to preprocessing service
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{PREPROCESSING_SERVICE_URL}/preprocess",
                json={"signal": ppg_signal.tolist()}
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Preprocessing complete: {result.get('num_segments')}")
                with state_lock:
                    collection_status = "COMPLETE"
                return result
            else:
                logger.error(f"Preprocessing failed: {response.text}")
                with state_lock:
                    collection_status = "ERROR"
                return None

    except Exception as e:
        logger.exception(f"Error processing data: {e}")
        with state_lock:
            collection_status = "ERROR"
        return None

@app.post("/stop")
async def stop_collection():
    """
    Stop data collection: send 'C' to ESP32 (like the BLE STOP control).
    """
    global connected_ws_app, collection_status

    with state_lock:
        if connected_ws_app is None:
            raise HTTPException(status_code=400, detail="Not connected")

    try:
        if connected_ws_app:
            connected_ws_app.send("C")
            logger.info("Sent STOP command ('C') to ESP32")
        with state_lock:
            collection_status = "STOPPED"
        return {"status": "stopped"}
    except Exception as e:
        logger.exception(f"Failed to stop collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disconnect")
async def disconnect_device():
    """
    Disconnect WebSocket client.
    """
    global connected_ws_app, ws_thread, ws_url_in_use, collection_status

    if connected_ws_app:
        try:
            connected_ws_app.close()
            # give thread a moment to finish
            time.sleep(0.2)
        except Exception as e:
            logger.exception(f"Error closing WebSocket: {e}")

    connected_ws_app = None
    ws_thread = None
    ws_url_in_use = None
    with state_lock:
        collection_status = "DISCONNECTED"
    logger.info("Disconnected from ESP32 WebSocket")
    return {"status": "disconnected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
