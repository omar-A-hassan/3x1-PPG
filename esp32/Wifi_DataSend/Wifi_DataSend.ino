/*
 * PPG Data Collector for Blood Glucose Prediction - WiFi HTTP Edition
 * ESP32 + MAX30102 Sensor
 *
 * Collects PPG data at ~50 Hz and transmits via HTTP POST to Python FastAPI server.
 * ESP32 acts as HTTP CLIENT, Python server receives the data.
 *
 * Flow:
 *   1. ESP32 waits for finger detection (autonomous)
 *   2. Collects 3000 samples (~60 seconds at 50 Hz)
 *   3. Transmits data chunks via HTTP POST to /receive_chunk
 *   4. Sends metadata to /collection_complete
 *   5. Returns to waiting state
 *
 * Hardware:
 *   - ESP32 DevKit (ESP32-WROOM-32)
 *   - MAX30102 Heart Rate & Pulse Oximeter Sensor
 */

#include <Wire.h>
#include "MAX30105.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ---------- CONFIGURATION ----------
#define I2C_SDA 21
#define I2C_SCL 22

#define SAMPLING_RATE 50      // Target sampling rate in Hz
#define TOTAL_SAMPLES 3000    // 60 seconds × 50 Hz = 3000 samples
#define SAMPLE_INTERVAL_US 20000  // 20,000 µs = 20 ms = 50 Hz

// WiFi Configuration
#define WIFI_MODE 0  // 0 = Connect to existing WiFi, 1 = ESP32 as Access Point

// If WIFI_MODE = 0: Connect to existing network
#define WIFI_SSID "WE_F49BD4"
#define WIFI_PASS "lcp20310"

// If WIFI_MODE = 1: ESP32 acts as Access Point
#define AP_SSID "ESP32-PPG-Glucose"

// Server Configuration - CHANGE THIS to your computer's IP address
#define SERVER_URL "http://192.168.1.6:8000"  // Your computer's IP on local WiFi
// ESP32 and computer must be on the same network (192.168.1.x)

#define SERVER_ENDPOINT "/receive_chunk"

// Quality thresholds
#define MIN_IR_THRESHOLD 12500
#define MAX_IR_THRESHOLD 200000

// Warmup time for sensor stabilization
const int WARMUP_MS = 3000;

// ---------- GLOBALS ----------
MAX30105 particleSensor;

// HTTP endpoints
String fullServerUrl;
String statusUrl;
String completeUrl;

// Collection Metadata (stored after collection for transmission)
float collection_metadata_duration = 0.0;
float collection_metadata_rate = 0.0;

// AP IP configuration
IPAddress local_IP(192, 168, 4, 1);

// Data storage
uint16_t ppgBuffer[TOTAL_SAMPLES];
uint32_t sampleIndex = 0;

// esp_timer for precise sampling
esp_timer_handle_t sampleTimer = NULL;
volatile bool sampleFlag = false;
volatile uint32_t isrTickCount = 0;
volatile uint64_t timerStartMicros = 0;

// ISR callback for timer
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
  WiFi.softAPConfig(local_IP, local_IP, IPAddress(255, 255, 255, 0));
  WiFi.softAP(AP_SSID);  // Open AP (no password)
  
  Serial.println("Access Point started!");
  Serial.print("SSID: ");
  Serial.println(AP_SSID);
  Serial.print("AP IP Address: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("\nConnect your computer to this WiFi network,");
  Serial.println("then update SERVER_URL with your computer's IP (192.168.4.x)");
#endif
}

// ---------- Send Status Update ----------
void sendStatus(const char* statusMsg) {
  HTTPClient http;
  http.begin(statusUrl);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);
  
  StaticJsonDocument<256> doc;
  doc["status"] = statusMsg;
  doc["timestamp"] = millis();
  doc["samples"] = samplesCollected;
  
  String jsonStr;
  serializeJson(doc, jsonStr);
  
  int httpCode = http.POST(jsonStr);
  if (httpCode > 0) {
    Serial.printf("Status sent: %s (HTTP %d)\n", statusMsg, httpCode);
  } else {
    Serial.printf("Status send failed: %s\n", http.errorToString(httpCode).c_str());
  }
  
  http.end();
}

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n\n");
  Serial.println("==============================================");
  Serial.println("PPG Data Collector - ESP32 + MAX30102");
  Serial.println("HTTP Client Edition (Posts to Python Server)");
  Serial.println("==============================================");

  // Build endpoint URLs
  fullServerUrl = String(SERVER_URL) + String(SERVER_ENDPOINT);
  statusUrl = String(SERVER_URL) + "/status_update";
  completeUrl = String(SERVER_URL) + "/collection_complete";
  
  Serial.print("Server URL: ");
  Serial.println(SERVER_URL);

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
  // sampleRate=400, sampleAverage=8 → 50 Hz effective output
  byte ledBrightness = 0x1F;
  byte sampleAverage = 8;
  byte ledMode = 2;        // Red + IR mode
  int sampleRate = 400;
  int pulseWidth = 411;
  int adcRange = 16384;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.clearFIFO();

  // Initialize WiFi
  initWiFi();

  // Test server availability
  delay(2000);
  HTTPClient http;
  http.begin(String(SERVER_URL) + "/");
  http.setTimeout(5000);
  int httpCode = http.GET();
  if (httpCode == 200) {
    Serial.println("✓ Server is reachable!");
  } else {
    Serial.println("⚠ Server not reachable. Will retry during operation.");
  }
  http.end();

  // Create esp_timer for precise sampling (50 Hz = 20 ms = 20,000 µs)
  const esp_timer_create_args_t timer_args = {
    .callback = &onSampleTimer,
    .arg = NULL,
    .dispatch_method = ESP_TIMER_TASK,
    .name = "sample_timer"
  };
  esp_err_t err = esp_timer_create(&timer_args, &sampleTimer);
  if (err != ESP_OK) {
    Serial.printf("ERROR: Failed to create timer: %d\n", err);
  } else {
    Serial.println("Timer created (50 Hz, starts on finger detect)");
  }

  Serial.println("\n✓ Ready! Place finger on sensor to begin.");
  Serial.println("State: IDLE → WAITING_FOR_FINGER");
}

// ---------- MAIN LOOP ----------
void loop() {
  switch (currentState) {
    case IDLE:
      // Auto-transition to waiting for finger
      currentState = WAITING_FOR_FINGER;
      sendStatus("WAITING");
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
      Serial.println("Collection cycle complete. Returning to IDLE...");
      delay(2000);
      currentState = IDLE;
      sendStatus("READY");
      break;
  }
}

// ---------- STATE: Waiting for Finger ----------
void handleWaitingForFinger() {
  uint32_t irValue = particleSensor.getIR();

  if (irValue > MIN_IR_THRESHOLD && irValue < MAX_IR_THRESHOLD) {
    Serial.println("Finger detected! Warming up sensor...");
    delay(WARMUP_MS);
    
    sendStatus("COLLECTING");
    
    // Reset collection state
    particleSensor.clearFIFO();
    sampleIndex = 0;
    samplesCollected = 0;
    poorQualitySamples = 0;
    sampleFlag = false;
    isrTickCount = 0;
    
    // Start timer
    if (sampleTimer) {
      timerStartMicros = esp_timer_get_time();
      esp_timer_start_periodic(sampleTimer, SAMPLE_INTERVAL_US);
      Serial.println("Timer started (50 Hz)");
    }
    
    currentState = COLLECTING;
    Serial.println("State: COLLECTING");
    
  } else {
    // Periodic status update
    static unsigned long lastStatusTime = 0;
    if (millis() - lastStatusTime > 2000) {
      if (irValue < MIN_IR_THRESHOLD) {
        Serial.println("Waiting for finger...");
      } else {
        Serial.println("Signal saturated - reduce pressure");
      }
      lastStatusTime = millis();
    }
  }
  
  delay(100);
}

// ---------- STATE: Collecting Samples ----------
void handleCollecting() {
  // Check if timer triggered a sample
  if (sampleFlag) {
    sampleFlag = false;
    
    // Read sensor
    uint32_t irValue = particleSensor.getIR();
    
    // Quality tracking
    if (irValue < MIN_IR_THRESHOLD || irValue > MAX_IR_THRESHOLD) {
      poorQualitySamples++;
      Serial.printf("Poor quality sample detected: IR=%u\n", irValue);
      currentState = IDLE;
      sendStatus("Restarting due to poor quality");
      break;
    }
    
    // Store sample (downscale from 18-bit to 16-bit by dividing by 4)
    ppgBuffer[sampleIndex] = (uint16_t)(irValue >> 2);
    sampleIndex++;
    samplesCollected++;
    
    // Check if collection complete
    if (sampleIndex >= TOTAL_SAMPLES) {
      // Stop timer
      if (sampleTimer) {
        esp_timer_stop(sampleTimer);
      }
      
      // Calculate precise timing
      uint64_t endMicros = esp_timer_get_time();
      float durationSec = (endMicros - timerStartMicros) / 1000000.0;
      float actualRate = (float)samplesCollected / durationSec;
      
      // Store for transmission
      collection_metadata_duration = durationSec;
      collection_metadata_rate = actualRate;
      
      Serial.println("\n========== Collection Complete ==========");
      Serial.printf("Samples: %d\n", samplesCollected);
      Serial.printf("Duration: %.2f seconds\n", durationSec);
      Serial.printf("Actual rate: %.2f Hz\n", actualRate);
      Serial.printf("Poor quality: %d (%.1f%%)\n", 
                    poorQualitySamples,
                    (float)poorQualitySamples / samplesCollected * 100.0);
      Serial.println("=========================================\n");
      
      sendStatus("TRANSMITTING");
      currentState = TRANSMITTING;
    }
  }
  
  // Diagnostic output every second
  static unsigned long lastDiag = 0;
  if (millis() - lastDiag >= 1000) {
    lastDiag = millis();
    
    uint64_t nowMicros = esp_timer_get_time();
    float elapsedSec = (nowMicros - timerStartMicros) / 1000000.0;
    float progress = (float)sampleIndex / TOTAL_SAMPLES * 100.0;
    
    Serial.printf("[Collecting] %u/%u samples (%.1f%%) | %.1fs elapsed\n",
                  sampleIndex, TOTAL_SAMPLES, progress, elapsedSec);
  }
  
  // Yield to WiFi stack
  delay(0);
}

// ---------- STATE: Transmitting Data ----------
void handleTransmitting() {
  Serial.println("Starting HTTP transmission...");

  const int SAMPLES_PER_CHUNK = 254;
  int totalChunks = (TOTAL_SAMPLES + SAMPLES_PER_CHUNK - 1) / SAMPLES_PER_CHUNK;

  HTTPClient http;
  
  for (int chunk = 0; chunk < totalChunks; chunk++) {
    int startIdx = chunk * SAMPLES_PER_CHUNK;
    int endIdx = min(startIdx + SAMPLES_PER_CHUNK, (int)TOTAL_SAMPLES);
    int chunkSamples = endIdx - startIdx;

    // Build binary packet: 4-byte header + sample data
    uint8_t packet[512];
    
    // Header (4 bytes)
    packet[0] = chunk & 0xFF;           // Chunk number (low byte)
    packet[1] = (chunk >> 8) & 0xFF;    // Chunk number (high byte)
    packet[2] = (uint8_t)totalChunks;   // Total chunks
    packet[3] = (uint8_t)chunkSamples;  // Samples in this chunk

    // Copy samples (little-endian uint16_t)
    memcpy(&packet[4], &ppgBuffer[startIdx], chunkSamples * sizeof(uint16_t));

    int packetSize = 4 + (chunkSamples * sizeof(uint16_t));

    // Send with retry logic
    bool sent = false;
    int retries = 3;
    
    for (int attempt = 0; attempt < retries && !sent; attempt++) {
      http.begin(fullServerUrl);
      http.addHeader("Content-Type", "application/octet-stream");
      http.addHeader("X-Chunk-Number", String(chunk));
      http.addHeader("X-Total-Chunks", String(totalChunks));
      http.setTimeout(15000);
      
      int httpCode = http.POST(packet, packetSize);
      
      if (httpCode == 200) {
        sent = true;
        float progress = (float)(chunk + 1) / totalChunks * 100.0;
        Serial.printf("Chunk %d/%d sent (%d samples, %.0f%%)\n",
                      chunk + 1, totalChunks, chunkSamples, progress);
      } else {
        Serial.printf("Chunk %d failed (attempt %d): HTTP %d\n", 
                      chunk + 1, attempt + 1, httpCode);
        if (attempt < retries - 1) {
          delay(1000);
        }
      }
      
      http.end();
    }
    
    if (!sent) {
      Serial.printf("ERROR: Failed to send chunk %d\n", chunk + 1);
      currentState = IDLE;
      return;
    }
    
    delay(50);  // Brief pause between chunks
  }

  // Send collection_complete with metadata
  http.begin(completeUrl);
  http.addHeader("Content-Type", "application/json");
  
  StaticJsonDocument<256> doc;
  doc["samples_collected"] = samplesCollected;
  doc["duration_seconds"] = collection_metadata_duration;
  doc["actual_sample_rate"] = collection_metadata_rate;
  doc["poor_quality_count"] = poorQualitySamples;
  
  String jsonStr;
  serializeJson(doc, jsonStr);
  
  int httpCode = http.POST(jsonStr);
  if (httpCode == 200) {
    Serial.println("✓ Collection metadata sent to server");
  } else {
    Serial.printf("⚠ Metadata send failed: HTTP %d\n", httpCode);
  }
  http.end();

  Serial.println("✓ Transmission complete!");
  currentState = COMPLETE;
}

// ---------- Diagnostics (optional) ----------
void printSensorDiagnostics() {
  Serial.println("\n=== MAX30102 Diagnostics ===");
  uint32_t irValue = particleSensor.getIR();
  uint32_t redValue = particleSensor.getRed();
  Serial.printf("IR: %u, Red: %u\n", irValue, redValue);
  Serial.printf("Temperature: %.2f°C\n", particleSensor.readTemperature());
  Serial.println("============================\n");
}
