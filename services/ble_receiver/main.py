"""
BLE Receiver Service
Receives PPG data from ESP32 via Bluetooth and forwards to preprocessing service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
import numpy as np
from bleak import BleakClient, BleakScanner
import struct
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BLE Receiver Service")

# Configuration
ESP32_DEVICE_NAME = "ESP32-PPG-Glucose"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
STATUS_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a9"
CONTROL_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26aa"

PREPROCESSING_SERVICE_URL = os.getenv("PREPROCESSING_SERVICE_URL", "http://localhost:8001")
TOTAL_SAMPLES = 6000

# Global state
connected_device = None
ppg_buffer = []
collection_status = "IDLE"

class CollectionRequest(BaseModel):
    action: str  # "start" or "stop"

class CollectionStatus(BaseModel):
    status: str
    samples_received: int
    total_samples: int

@app.get("/")
async def root():
    return {
        "service": "BLE Receiver",
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
    """Scan for ESP32 device"""
    logger.info("Scanning for BLE devices...")

    devices = await BleakScanner.discover(timeout=5.0)
    esp32_devices = [
        {"name": d.name, "address": d.address, "rssi": d.rssi}
        for d in devices
        if d.name and ESP32_DEVICE_NAME in d.name
    ]

    logger.info(f"Found {len(esp32_devices)} ESP32 devices")
    return {"devices": esp32_devices}

@app.post("/connect")
async def connect_device(device_address: str):
    """Connect to ESP32 device"""
    global connected_device, collection_status

    try:
        logger.info(f"Connecting to {device_address}...")
        client = BleakClient(device_address)
        await client.connect()

        connected_device = client
        collection_status = "CONNECTED"

        logger.info("Connected successfully")
        return {"status": "connected", "address": device_address}

    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect")
async def start_collection():
    """Start PPG data collection"""
    global ppg_buffer, collection_status

    if not connected_device or not connected_device.is_connected:
        raise HTTPException(status_code=400, detail="No device connected")

    try:
        # Reset buffer
        ppg_buffer = []
        collection_status = "COLLECTING"

        # Setup notification handler
        await connected_device.start_notify(
            DATA_CHAR_UUID,
            data_notification_handler
        )

        await connected_device.start_notify(
            STATUS_CHAR_UUID,
            status_notification_handler
        )

        # Send START command
        await connected_device.write_gatt_char(
            CONTROL_CHAR_UUID,
            b'S'
        )

        logger.info("Collection started")
        return {"status": "collecting"}

    except Exception as e:
        logger.error(f"Failed to start collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def data_notification_handler(sender, data: bytearray):
    """Handle incoming PPG data chunks"""
    global ppg_buffer

    try:
        # Parse chunk header: [chunk_num(2), total_chunks(1), sample_count(1), ...data...]
        chunk_num = struct.unpack('<H', data[0:2])[0]
        total_chunks = data[2]
        sample_count = data[3]

        # Extract samples (uint16, little-endian)
        samples = []
        for i in range(sample_count):
            offset = 4 + i * 2
            sample = struct.unpack('<H', data[offset:offset+2])[0]
            samples.append(sample)

        ppg_buffer.extend(samples)

        logger.info(f"Received chunk {chunk_num + 1}/{total_chunks}: {len(samples)} samples (total: {len(ppg_buffer)})")

        # If collection complete, process data
        if len(ppg_buffer) >= TOTAL_SAMPLES:
            asyncio.create_task(process_collected_data())

    except Exception as e:
        logger.error(f"Error processing data chunk: {e}")

def status_notification_handler(sender, data: bytearray):
    """Handle status updates from ESP32"""
    global collection_status

    status_msg = data.decode('utf-8')
    collection_status = status_msg
    logger.info(f"ESP32 Status: {status_msg}")

async def process_collected_data():
    """Send collected data to preprocessing service"""
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
                logger.info(f"Preprocessing complete: {result['num_segments']} segments")
                collection_status = "COMPLETE"
                return result
            else:
                logger.error(f"Preprocessing failed: {response.text}")
                collection_status = "ERROR"
                return None

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        collection_status = "ERROR"
        return None

@app.post("/stop")
async def stop_collection():
    """Stop data collection"""
    global collection_status

    if not connected_device or not connected_device.is_connected:
        raise HTTPException(status_code=400, detail="No device connected")

    try:
        # Send STOP command
        await connected_device.write_gatt_char(
            CONTROL_CHAR_UUID,
            b'C'
        )

        collection_status = "STOPPED"
        logger.info("Collection stopped")
        return {"status": "stopped"}

    except Exception as e:
        logger.error(f"Failed to stop collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/disconnect")
async def disconnect_device():
    """Disconnect from ESP32"""
    global connected_device, collection_status

    if connected_device and connected_device.is_connected:
        await connected_device.disconnect()
        logger.info("Disconnected")

    connected_device = None
    collection_status = "DISCONNECTED"
    return {"status": "disconnected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
