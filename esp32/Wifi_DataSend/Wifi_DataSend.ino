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

#define SAMPLING_RATE 200  
#define COLLECTION_TIME 100  // 90 seconds for 9000 samples
#define TOTAL_SAMPLES 3000  // Fixed target: 9000 samples
#define SAMPLE_INTERVAL_MS (1000 / SAMPLING_RATE)

// WiFi Configuration (Choose one)
#define WIFI_MODE 1  // 0 = Connect to existing WiFi, 1 = ESP32 as Access Point

// If WIFI_MODE = 0: Connect to existing network
#define WIFI_SSID "ESP32GlucoLevel"
#define WIFI_PASS "12345678"

// If WIFI_MODE = 1: ESP32 acts as Access Point
#define AP_SSID "ESP32-PPG-Glucose"
// For initial bring-up, use OPEN AP to avoid CCMP replay issues
// #define AP_PASS "ESP32gluco123"

#define WEBSERVER_PORT 80
#define WEBSOCKET_PORT 81

// Quality thresholds
#define MIN_IR_THRESHOLD 12500
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
unsigned long lastSampleTime = 0; // legacy; not used with timer

// esp_timer for precise sampling (ESP-IDF native timer API)
esp_timer_handle_t sampleTimer = NULL;
volatile bool sampleFlag = false;
volatile uint32_t isrTickCount = 0;
volatile uint64_t timerStartMicros = 0;  // Track actual start time for duration
const int warmup_ms = 3000;  // Wait till IR readings stabilize


void IRAM_ATTR onSampleTimer(void* arg) {
  sampleFlag = true;
  isrTickCount++;
}

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
    // Open AP to reduce WPA2/CCMP issues during testing
    // WiFi.softAP(AP_SSID, AP_PASS,6,false);
    WiFi.softAP(AP_SSID);
    
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
              // Stop timer if active
              if (sampleTimer) {
                esp_timer_stop(sampleTimer);
              }
              sendStatus("CANCELLED");
              break;
              
            case 'R':  // RESET
              Serial.println("Command: RESET");
              currentState = IDLE;
              sampleIndex = 0;
              samplesCollected = 0;
              // Stop timer if active
              if (sampleTimer) {
                esp_timer_stop(sampleTimer);
              }
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

  // Configure MAX30102 for optimal signal quality at 100 Hz
  // Config C: sampleRate=400 Hz, sampleAverage=4 → 100 Hz output (better for glucose segmentation)
  byte ledBrightness = 0x1F;   // Options: 0=Off to 255=50mA
  byte sampleAverage = 4;      // Hardware averaging: 4 samples @ 400 Hz → 100 Hz output (~6dB SNR)
  byte ledMode = 2;            // Red + IR mode
  int sampleRate = 200;        // Sensor ADC samples at 400 Hz, averages to 100 Hz
  int pulseWidth = 411;        // 411 µs LED pulse (good SNR without saturation)
  int adcRange = 16384;        // 16-bit ADC range

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.clearFIFO();

  // Initialize WiFi
  initWiFi();

  // Initialize WebSocket
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.printf("WebSocket server started on port %d\n", WEBSOCKET_PORT);

  Serial.println("\nReady! Waiting for client connection...");
  Serial.println("State: IDLE");

  // Create esp_timer for 100 Hz sampling (10 ms period = 10,000 µs)
  const esp_timer_create_args_t timer_args = {
    .callback = &onSampleTimer,
    .arg = NULL,
    .dispatch_method = ESP_TIMER_TASK,
    .name = "sample_timer"
  };
  esp_err_t err = esp_timer_create(&timer_args, &sampleTimer);
  if (err != ESP_OK) {
    Serial.printf("ERROR: Failed to create esp_timer: %d\n", err);
  } else {
    Serial.println("esp_timer created successfully (100 Hz = 10 ms period, will start on finger detect)");
  }
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
    Serial.println("Warming up sensor please wait...");
    delay(warmup_ms); 
    sendStatus("COLLECTING");
    currentState = COLLECTING;
    particleSensor.clearFIFO();
    collectionStartTime = millis();
    sampleIndex = 0;
    samplesCollected = 0;
    poorQualitySamples = 0;
    lastSampleTime = millis();
    // Start esp_timer for precise 100 Hz sampling
    sampleFlag = false;
    isrTickCount = 0;
    if (sampleTimer) {
      timerStartMicros = esp_timer_get_time();  // Record start in microseconds
      esp_timer_start_periodic(sampleTimer, 10000);  // 10,000 µs = 10 ms = 100 Hz
      Serial.println("[Diag] esp_timer started (100 Hz periodic)");
    }
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
  // Use esp_timer-driven flag to trigger sampling at precise intervals
  bool doSample = false;
  if (sampleFlag) {
    sampleFlag = false;
    doSample = true;
  }

  // Periodic diagnostics to Serial ONLY (no WebSocket overhead)
  static unsigned long lastDiag = 0;
  static uint32_t lastTickCount = 0;
  unsigned long now = millis();
  if (now - lastDiag >= 1000) {
    uint32_t ticks = isrTickCount;
    uint32_t delta = ticks - lastTickCount;
    lastTickCount = ticks;
    lastDiag = now;
    
    // Show elapsed time and effective rate from esp_timer
    uint64_t nowMicros = esp_timer_get_time();
    float elapsedSec = (nowMicros - timerStartMicros) / 1000000.0;
    float effectiveRate = sampleIndex > 0 ? sampleIndex / elapsedSec : 0.0;
    float progress = (float)sampleIndex / TOTAL_SAMPLES * 100.0;
    
    Serial.printf("[Diag] ISR: %u ticks/s | Samples: %u/%u (%.1f%%) | Elapsed: %.2fs | Rate: %.2f Hz\n", 
                  delta, sampleIndex, TOTAL_SAMPLES, progress, elapsedSec, effectiveRate);
    
    // Warn if timer appears stalled
    if (delta == 0) {
      Serial.println("[WARNING] No ISR ticks in last second - timer may be stalled!");
    }
  }

  if (doSample) {
    // Read sensor value
    uint32_t irValue = particleSensor.getIR();

    // Quality tracking (statistics only, no output)
    bool goodQuality = (irValue > MIN_IR_THRESHOLD && irValue < MAX_IR_THRESHOLD);
    if (!goodQuality) {
      poorQualitySamples++;
    }

    // Store sample (downscale from 18-bit to 16-bit)
    ppgBuffer[sampleIndex] = (uint16_t)(irValue >> 2);
    sampleIndex++;
    samplesCollected++;

    // NO PROGRESS UPDATES - Removed to maximize sampling rate
    // Previously sent JSON every 10 samples, causing ~30-40ms overhead per iteration
    // Now the loop can run at nearly 100 Hz without WebSocket transmission blocking

    // Check if collection complete
    if (sampleIndex >= TOTAL_SAMPLES) {
      // Stop timer and compute precise duration from esp_timer
      if (sampleTimer) {
        esp_timer_stop(sampleTimer);
      }
      uint64_t endMicros = esp_timer_get_time();
      float collectionDuration = (endMicros - timerStartMicros) / 1000000.0;  // seconds
      float actualRate = (float)samplesCollected / collectionDuration;

      Serial.println("\n=== Collection Complete ===");
      Serial.printf("Samples collected: %d\n", samplesCollected);
      Serial.printf("Duration: %.2f seconds\n", collectionDuration);
      Serial.printf("Actual rate: %.2f Hz\n", actualRate);
      Serial.printf("Poor quality samples: %d (%.2f%%)\n",
                    poorQualitySamples,
                    (float)poorQualitySamples / samplesCollected * 100.0);
      Serial.println("==========================\n");

      // Send timing metadata to Python
      StaticJsonDocument<256> doc;
      doc["event"] = "collection_complete";
      doc["samples_collected"] = samplesCollected;
      doc["duration_seconds"] = collectionDuration;
      doc["actual_sample_rate"] = actualRate;
      doc["poor_quality_count"] = poorQualitySamples;
      
      String jsonStr;
      serializeJson(doc, jsonStr);
      webSocket.sendTXT(connectedClientNum, jsonStr);
      Serial.println("Sent timing metadata to Python");

      sendStatus("TRANSMITTING");
      currentState = TRANSMITTING;
    }
  }

  // Minimal yield to WiFi stack (required to prevent watchdog timeout)
  // delay(0) calls yield() internally - sufficient for WiFi keepalive
  delay(0);
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

  Serial.printf("Die Temperature: %.2f°C\n", particleSensor.readTemperature());
  Serial.println("===========================\n");
}