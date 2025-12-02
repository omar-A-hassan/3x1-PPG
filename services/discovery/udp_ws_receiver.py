"""
Standalone UDP discovery + WebSocket receiver
- Listens for ESP32 UDP broadcast on 255.255.255.255:19532
- Builds ws://<ip>:<port>/ and connects
- Prints incoming text/binary messages and exits on completion

Run:
  python services/discovery/udp_ws_receiver.py

Env (optional):
  DISCOVERY_PORT=19532
  ESP32_DEVICE_NAME=ESP32-PPG-Glucose
  WS_FALLBACK_URL=ws://192.168.4.1:81/
"""

import os
import socket
import json
import time
import struct
import logging
from websocket import create_connection

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("udp_ws_receiver")

DISCOVERY_PORT = int(os.getenv("DISCOVERY_PORT", "19532"))
DEVICE_NAME = os.getenv("ESP32_DEVICE_NAME", "ESP32-PPG-Glucose")
FALLBACK_URL = os.getenv("WS_FALLBACK_URL", "ws://192.168.4.1:81/")


def discover_once(timeout=5.0):
    """Listen for a single UDP broadcast packet and return ws URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("0.0.0.0", DISCOVERY_PORT))
        sock.settimeout(timeout)
        log.info(f"Listening for UDP discovery on 0.0.0.0:{DISCOVERY_PORT} (timeout {timeout}s)")
        data, addr = sock.recvfrom(1024)
        payload = data.decode("utf-8", errors="ignore")
        log.info(f"Discovery packet from {addr}: {payload}")
        js = json.loads(payload)
        if js.get("device") != DEVICE_NAME:
            raise RuntimeError("Device name mismatch")
        ip = js.get("ip")
        port = int(js.get("ws_port", 81))
        url = f"ws://{ip}:{port}/"
        return url
    except socket.timeout:
        log.warning("Discovery timeout; using fallback URL")
        return FALLBACK_URL
    finally:
        sock.close()


def connect_and_run(url: str):
    log.info(f"Connecting to WebSocket: {url}")
    ws = create_connection(url, timeout=5)
    try:
        # Ask device for status, then start if desired
        ws.send("G")  # GET STATUS
        log.info("Sent 'G' for status")
        # Example: start a collection
        ws.send("S")
        log.info("Sent 'S' to start collection")

        while True:
            msg = ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                log.info(f"BINARY {len(msg)} bytes")
                # Optional: parse header like in Wifi_main_csv.py
                if len(msg) >= 4:
                    chunk_num = struct.unpack('<H', msg[0:2])[0]
                    total_chunks = msg[2]
                    sample_count = msg[3]
                    log.info(f"Chunk {chunk_num+1}/{total_chunks} samples={sample_count}")
            else:
                log.info(f"TEXT: {msg}")
                if isinstance(msg, str) and msg.strip().startswith('{'):
                    try:
                        js = json.loads(msg)
                        if js.get("event") == "collection_complete":
                            log.info("Collection complete metadata received; exiting")
                            break
                    except Exception:
                        pass
    finally:
        ws.close()
        log.info("WebSocket closed")


if __name__ == "__main__":
    url = discover_once(timeout=5.0)
    print(f"Using WebSocket URL: {url}")
    connect_and_run(url)
