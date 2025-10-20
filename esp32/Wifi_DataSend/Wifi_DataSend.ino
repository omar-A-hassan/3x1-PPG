/*
 * PPG Data Collector for Blood Glucose Prediction - WiFi Edition
 * ESP32 + MAX30102 Sensor
 *
 * Collects 1 minute of PPG data at 100 Hz and transmits via WiFi to iOS app
 * Uses ESP32 as WebSocket server or REST API endpoint
 *
 * Hardware:
 *   - ESP32 DevKit (ESP32-WROOM-32)
 *   - MAX30102 Heart Rate & Pulse Oximeter Sensor
 *
 * WiFi Options:
 *   Option A: ESP32 connects to existing WiFi network
 *   Option B: ESP32 acts as Access Point (AP) - iOS connects directly
 */

#include <Wire.h>
#include "MAX30105.h"
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

// ---------- CONFIGURATION ----------
#define I2C_SDA 21
#define I2C_SCL 22

#define SAMPLING_RATE 100
#define COLLECTION_TIME 60
#define TOTAL_SAMPLES (SAMPLING_RATE * COLLECTION_TIME)  // 6000 samples
#define SAMPLE_INTERVAL_MS (1000 / SAMPLING_RATE)

// WiFi Configuration (Choose one)
#define WIFI_MODE 1  // 0 = Connect to existing WiFi, 1 = ESP32 as Access Point

// If WIFI_MODE = 0: Connect to existing network
#define WIFI_SSID "ESP32GlucoLevel"
#define WIFI_PASS "12345678"

// If WIFI_MODE = 1: ESP32 acts as Access Point
#define AP_SSID "ESP32-PPG-Glucose"
#define AP_PASS "ESP32gluco123"

#define WEBSERVER_PORT 80
#define WEBSOCKET_PORT 81

// Quality thresholds
#define MIN_IR_THRESHOLD 50000
#define MAX_IR_THRESHOLD 200000

// ---------- GLOBALS ----------
MAX30105 particleSensor;

// WiFi & WebSocket
WebSocketsServer webSocket = WebSocketsServer(WEBSOCKET_PORT);
bool clientConnected = false;
uint8_t connectedClientNum = 0;

IPAddress local_IP(192,168,4,1);

// Data storage
uint16_t ppgBuffer[TOTAL_SAMPLES];
uint32_t sampleIndex = 0;
unsigned long lastSampleTime = 0;

// State machine
enum State {
  IDLE,
  WAITING_FOR_FINGER,
  COLLECTING,
  TRANSMITTING,
  COMPLETE
};
State currentState = IDLE;

// Statistics
uint32_t samplesCollected = 0;
uint32_t poorQualitySamples = 0;
unsigned long collectionStartTime = 0;

// ---------- WiFi Setup ----------
void initWiFi() {
  #if WIFI_MODE == 0
    // Connect to existing WiFi network
    Serial.println("Connecting to WiFi...");
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\nWiFi connected!");
      Serial.print("IP Address: ");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("\nFailed to connect to WiFi");
    }
  
  #elif WIFI_MODE == 1
    // ESP32 acts as Access Point
    Serial.println("Starting WiFi Access Point...");
    WiFi.mode(WIFI_AP);
    WiFi.mode(WIFI_AP);
    WiFi.softAPConfig(local_IP, local_IP, IPAddress(255,255,255,0)); // gateway, subnet
    WiFi.softAP(AP_SSID, AP_PASS,6,false);
    
    IPAddress apIP = WiFi.softAPIP();
    Serial.println("Access Point started!");
    Serial.print("SSID: ");
    Serial.println(AP_SSID);
    Serial.print("IP Address: ");
    Serial.println(apIP);
  #endif
}

// ---------- WebSocket Callbacks ----------
void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_CONNECTED:
      {
        Serial.printf("WebSocket client #%u connected\n", num);
        connectedClientNum = num;
        clientConnected = true;
        
        // Send welcome message
        String welcome = "{\"event\":\"connected\",\"version\":\"1.0\"}";
        webSocket.sendTXT(num, welcome);
        
        sendStatus("READY");
      }
      break;
      
    case WStype_DISCONNECTED:
      {
        Serial.printf("WebSocket client #%u disconnected\n", num);
        clientConnected = false;
        currentState = IDLE;
      }
      break;
      
    case WStype_TEXT:
      {
        // Parse incoming command
        Serial.printf("Received command from client #%u: ", num);
        for(size_t i = 0; i < length; i++) {
          Serial.print((char)payload[i]);
        }
        Serial.println();
        
        // Simple command parsing (single character)
        if (length > 0) {
          char command = (char)payload[0];
          
          switch (command) {
            case 'S':  // START
              if (currentState == IDLE) {
                Serial.println("Command: START collection");
                currentState = WAITING_FOR_FINGER;
                sampleIndex = 0;
                samplesCollected = 0;
                poorQualitySamples = 0;
                sendStatus("WAITING");
              }
              break;
              
            case 'C':  // CANCEL
              Serial.println("Command: CANCEL");
              currentState = IDLE;
              sampleIndex = 0;
              sendStatus("CANCELLED");
              break;
              
            case 'R':  // RESET
              Serial.println("Command: RESET");
              currentState = IDLE;
              sampleIndex = 0;
              samplesCollected = 0;
              sendStatus("RESET");
              break;
              
            case 'G':  // GET STATUS
              sendStatus("STATUS_REQUEST");
              break;
          }
        }
      }
      break;
      
    case WStype_BIN:
      Serial.println("Binary data received");
      break;
  }
}

// ---------- STATUS & COMMAND ----------
void sendStatus(const char* statusMsg) {
  if (!clientConnected) return;
  
  // Create JSON status message
  StaticJsonDocument<256> doc;
  doc["event"] = "status";
  doc["status"] = statusMsg;
  doc["timestamp"] = millis();
  doc["state"] = currentState;
  doc["samples"] = samplesCollected;
  
  String jsonStr;
  serializeJson(doc, jsonStr);
  
  webSocket.sendTXT(connectedClientNum, jsonStr);
  Serial.printf("Status sent: %s\n", statusMsg);
}

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  delay(2000);  // Wait for serial monitor
  
  Serial.println("\n\n");
  Serial.println("PPG Data Collector - ESP32 + MAX30102 (WiFi Edition)");
  Serial.println("=====================================================");

  // Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  // Initialize MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found!");
    while (1) { delay(1000); }
  }
  Serial.println("MAX30102 initialized");

  // Configure MAX30102
  byte ledBrightness = 0x1F;
  byte sampleAverage = 4;
  byte ledMode = 2;
  int sampleRate = 100;
  int pulseWidth = 411;
  int adcRange = 16384;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.clearFIFO();

  Serial.println("Sensor configured for 100 Hz");

  // Initialize WiFi
  initWiFi();

  // Initialize WebSocket
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.printf("WebSocket server started on port %d\n", WEBSOCKET_PORT);

  Serial.println("\nReady! Waiting for client connection...");
  Serial.println("State: IDLE");
}

// ---------- MAIN LOOP ----------
void loop() {
  // Handle WebSocket communication
  webSocket.loop();

  // State machine
  switch (currentState) {
    case IDLE:
      delay(100);
      break;

    case WAITING_FOR_FINGER:
      handleWaitingForFinger();
      break;

    case COLLECTING:
      handleCollecting();
      break;

    case TRANSMITTING:
      handleTransmitting();
      break;

    case COMPLETE:
      delay(1000);
      currentState = IDLE;
      sendStatus("READY");
      Serial.println("State: IDLE");
      break;
  }
}

// ---------- STATE HANDLERS ----------
void handleWaitingForFinger() {
  uint32_t irValue = particleSensor.getIR();

  if (irValue > MIN_IR_THRESHOLD && irValue < MAX_IR_THRESHOLD) {
    Serial.println("Finger detected! Starting collection...");
    sendStatus("COLLECTING");
    currentState = COLLECTING;
    collectionStartTime = millis();
    sampleIndex = 0;
    samplesCollected = 0;
    poorQualitySamples = 0;
    lastSampleTime = millis();
  } else {
    static unsigned long lastStatusTime = 0;
    if (millis() - lastStatusTime > 1000) {
      if (irValue < MIN_IR_THRESHOLD) {
        sendStatus("PLACE_FINGER");
        Serial.println("Waiting for finger...");
      } else {
        sendStatus("ADJUST_PRESSURE");
        Serial.println("Signal saturated - reduce pressure");
      }
      lastStatusTime = millis();
    }
    delay(100);
  }
}

void handleCollecting() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    uint32_t irValue = particleSensor.getIR();

    bool goodQuality = (irValue > MIN_IR_THRESHOLD && irValue < MAX_IR_THRESHOLD);
    if (!goodQuality) {
      poorQualitySamples++;

      if (poorQualitySamples > 100) {
        Serial.println("ERROR: Poor signal quality - collection aborted");
        sendStatus("ERROR_QUALITY");
        currentState = IDLE;
        return;
      }
    }

    ppgBuffer[sampleIndex] = (uint16_t)(irValue >> 2);
    sampleIndex++;
    samplesCollected++;

    // Progress updates every 10 samples
    if (sampleIndex % 10 == 0) {
      float progress = (float)sampleIndex / TOTAL_SAMPLES * 100.0;
      
      // Send progress as JSON
      StaticJsonDocument<128> doc;
      doc["event"] = "progress";
      doc["progress"] = progress;
      doc["samples"] = sampleIndex;
      
      String jsonStr;
      serializeJson(doc, jsonStr);
      webSocket.sendTXT(connectedClientNum, jsonStr);

      if (sampleIndex % 100 == 0) {
        Serial.printf("Progress: %.1f%% (%d/%d samples)\n", progress, sampleIndex, TOTAL_SAMPLES);
      }
    }

    // Check if collection complete
    if (sampleIndex >= TOTAL_SAMPLES) {
      unsigned long collectionDuration = millis() - collectionStartTime;
      float actualRate = (float)samplesCollected / (collectionDuration / 1000.0);

      Serial.println("\n=== Collection Complete ===");
      Serial.printf("Samples collected: %d\n", samplesCollected);
      Serial.printf("Duration: %.2f seconds\n", collectionDuration / 1000.0);
      Serial.printf("Actual rate: %.2f Hz\n", actualRate);
      Serial.printf("Poor quality samples: %d (%.2f%%)\n",
                    poorQualitySamples,
                    (float)poorQualitySamples / samplesCollected * 100.0);
      Serial.println("==========================\n");

      sendStatus("TRANSMITTING");
      currentState = TRANSMITTING;
    }
  }
}

void handleTransmitting() {
  if (!clientConnected) {
    Serial.println("ERROR: Client disconnected during transmission");
    currentState = IDLE;
    return;
  }

  Serial.println("Starting data transmission via WiFi...");

  // WiFi allows larger packets - send 512 bytes at a time
  // Each packet: 4-byte header + 254 samples × 2 bytes = 512 bytes
  const int SAMPLES_PER_CHUNK = 254;
  int totalChunks = (TOTAL_SAMPLES + SAMPLES_PER_CHUNK - 1) / SAMPLES_PER_CHUNK;

  for (int chunk = 0; chunk < totalChunks; chunk++) {
    int startIdx = chunk * SAMPLES_PER_CHUNK;
    int endIdx = min(startIdx + SAMPLES_PER_CHUNK, (int)TOTAL_SAMPLES);
    int chunkSamples = endIdx - startIdx;

    // Build binary packet with header
    uint8_t packet[512];
    
    // Header (4 bytes) - JSON-friendly format
    packet[0] = chunk & 0xFF;              // chunk_num low byte
    packet[1] = (chunk >> 8) & 0xFF;       // chunk_num high byte
    packet[2] = (uint8_t)totalChunks;      // total_chunks
    packet[3] = (uint8_t)chunkSamples;     // sample_count

    // Copy samples (little-endian)
    memcpy(&packet[4], &ppgBuffer[startIdx], chunkSamples * sizeof(uint16_t));

    // Calculate packet size
    int packetSize = 4 + (chunkSamples * sizeof(uint16_t));

    // Send binary data via WebSocket
    webSocket.sendBIN(connectedClientNum, packet, packetSize);

    // Progress update
    float progress = (float)(chunk + 1) / totalChunks * 100.0;
    
    StaticJsonDocument<128> doc;
    doc["event"] = "transmission";
    doc["chunk"] = chunk + 1;
    doc["total_chunks"] = totalChunks;
    doc["progress"] = progress;
    
    String jsonStr;
    serializeJson(doc, jsonStr);
    webSocket.sendTXT(connectedClientNum, jsonStr);

    Serial.printf("Transmitted chunk %d/%d: %d samples, %d bytes (%.1f%%)\n",
                  chunk + 1, totalChunks, chunkSamples, packetSize, progress);

    // Much faster than BLE - no delay needed, WiFi handles buffering
    delay(5);  // Small delay to let WebSocket process
  }

  Serial.println("Transmission complete!");
  sendStatus("COMPLETE");
  currentState = COMPLETE;
}

// ---------- OPTIONAL: Diagnostics ----------
void printSensorDiagnostics() {
  Serial.println("\n=== MAX30102 Diagnostics ===");
  uint32_t irValue = particleSensor.getIR();
  uint32_t redValue = particleSensor.getRed();
  Serial.printf("IR Value: %d, Red Value : %d\n", irValue,redValue);
  Serial.println(irValue);

  if (irValue < MIN_IR_THRESHOLD) {
    Serial.println("Status: NO FINGER DETECTED");
  } else if (irValue > MAX_IR_THRESHOLD) {
    Serial.println("Status: SIGNAL SATURATED (too much pressure)");
  } else {
    Serial.println("Status: GOOD SIGNAL");
  }
  Serial.printf("Die Temperature: %.2f°C\n", particleSensor.readTemperature());
  Serial.println("===========================\n");
}