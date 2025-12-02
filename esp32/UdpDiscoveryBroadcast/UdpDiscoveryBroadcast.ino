/*
 * ESP32 UDP Discovery + WebSocket Server (Standalone)
 * - STA mode: connects to existing WiFi (hotspot)
 * - Broadcasts UDP JSON to 255.255.255.255:19532 with device info
 * - Runs WebSocket server on port 81, accepts 'S','C','G' commands
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

// ---- CONFIG ----
#define WIFI_SSID "Speed Tuner"
#define WIFI_PASS "Lol123456"
#define DEVICE_NAME "ESP32-PPG-Glucose"
#define WS_PORT 81
#define DISCOVERY_PORT 19532

// ---- Globals ----
WebSocketsServer webSocket = WebSocketsServer(WS_PORT);
WiFiUDP udp;
bool clientConnected = false;
uint8_t connectedClientNum = 0;
unsigned long lastBroadcast = 0;

// ---- WiFi ----
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Connecting to WiFi SSID=%s\n", WIFI_SSID);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 80) {
    delay(500);
    Serial.print('.');
    attempts++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("WiFi connected, IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("Failed to connect to WiFi");
  }
}

// ---- UDP Discovery ----
void sendDiscovery() {
  if (WiFi.status() != WL_CONNECTED) return;
  StaticJsonDocument<192> doc;
  doc["device"] = DEVICE_NAME;
  doc["ip"] = WiFi.localIP().toString();
  doc["ws_port"] = WS_PORT;
  doc["mac"] = WiFi.macAddress();
  char buf[192];
  size_t len = serializeJson(doc, buf);
  udp.beginPacket(IPAddress(255,255,255,255), DISCOVERY_PORT);
  udp.write((uint8_t*)buf, len);
  udp.endPacket();
  Serial.printf("Discovery: %s\n", buf);
}

// ---- WebSocket ----
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_CONNECTED: {
      connectedClientNum = num;
      clientConnected = true;
      Serial.printf("Client #%u connected\n", num);
      StaticJsonDocument<128> doc;
      doc["event"] = "status";
      doc["status"] = "READY";
      String welcome; serializeJson(doc, welcome);
      webSocket.sendTXT(num, welcome);
    } break;
    case WStype_DISCONNECTED: {
      Serial.printf("Client #%u disconnected\n", num);
      clientConnected = false;
    } break;
    case WStype_TEXT: {
      char cmd = (length > 0) ? (char)payload[0] : '\0';
      Serial.printf("Cmd from #%u: %c\n", num, cmd);
      if (cmd == 'G') {
        StaticJsonDocument<128> doc;
        doc["event"] = "status";
        doc["status"] = "READY";
        String s; serializeJson(doc, s);
        webSocket.sendTXT(num, s);
      } else if (cmd == 'S') {
        // Demo: send a small binary frame (header + few samples)
        const int samples = 10;
        uint8_t packet[4 + samples*2];
        packet[0] = 0; packet[1] = 0; // chunk_num
        packet[2] = 1;                // total_chunks
        packet[3] = samples;          // sample_count
        for (int i=0;i<samples;i++) {
          uint16_t v = (uint16_t)(1000 + i);
          packet[4 + i*2] = v & 0xFF;
          packet[5 + i*2] = (v >> 8) & 0xFF;
        }
        webSocket.sendBIN(num, packet, sizeof(packet));
        StaticJsonDocument<128> cdoc;
        cdoc["event"] = "collection_complete";
        cdoc["samples_collected"] = samples;
        cdoc["duration_seconds"] = 0.1;
        cdoc["actual_sample_rate"] = 100.0;
        String js; serializeJson(cdoc, js);
        webSocket.sendTXT(num, js);
      } else if (cmd == 'C') {
        StaticJsonDocument<96> doc;
        doc["event"] = "status";
        doc["status"] = "CANCELLED";
        String s; serializeJson(doc, s);
        webSocket.sendTXT(num, s);
      }
    } break;
    default: break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  connectWiFi();
  udp.begin(0); // port 0 for outbound only
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.printf("WebSocket server on port %d\n", WS_PORT);
}

void loop() {
  webSocket.loop();
  unsigned long now = millis();
  if (now - lastBroadcast > 2000) {
    sendDiscovery();
    lastBroadcast = now;
  }
}
