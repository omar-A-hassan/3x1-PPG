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
#include <WebServer.h>
#include <Preferences.h>
#include <LiquidCrystal_I2C.h>

// ---------- CONFIGURATION ----------
#define I2C_SDA 21
#define I2C_SCL 22

// ECG input and lead-off pins
#define ECG_PIN 35       // ADC1_CH7 (ECG signal input)
#define LO_PLUS_PIN 25   // Lead-off detection + (safe GPIO)
#define LO_MINUS_PIN 26  // Lead-off detection - (safe GPIO)

#define SAMPLING_RATE 50      // Target sampling rate in Hz
#define TOTAL_SAMPLES 3000    // 60 seconds × 50 Hz = 3000 samples
#define SAMPLE_INTERVAL_US 20000  // 20,000 µs = 20 ms = 50 Hz

// WiFi Configuration - Now uses web-based provisioning
#define AP_SSID "ESP32-PPG-Setup"  // AP name for configuration mode
#define RESET_BUTTON_PIN 0          // BOOT button on most ESP32 boards

// Server endpoint (appended to saved server URL)
#define SERVER_ENDPOINT "/receive_chunk"

// Quality thresholds
#define MIN_IR_THRESHOLD 12500
#define MAX_IR_THRESHOLD 200000

// Warmup time for sensor stabilization
const int WARMUP_MS = 3000;

// ---------- GLOBALS ----------
MAX30105 particleSensor;
LiquidCrystal_I2C lcd(0x27, 16, 2);
bool serverConnected = false;

// Web server for configuration
WebServer configServer(80);
Preferences preferences;

// Saved configuration (loaded from Preferences)
String savedSSID = "";
String savedPassword = "";
String savedServerUrl = "";
String savedApiKey = "";

// HTTP endpoints (built from savedServerUrl)
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
uint16_t ecgBuffer[TOTAL_SAMPLES];
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
  CONFIGURE_WIFI,      // AP mode + web server for provisioning
  CONNECTING_WIFI,     // Attempting to connect with saved credentials
  IDLE,
  WAITING_FOR_FINGER,
  COLLECTING,
  TRANSMITTING,
  COMPLETE
};
State currentState = CONFIGURE_WIFI;

// Statistics
uint32_t samplesCollected = 0;
uint32_t poorQualitySamples = 0;
unsigned long collectionStartTime = 0;

// ---------- Forward Declarations ----------
void handleConfigureWiFi();
void handleConnectingWiFi();
void handleWaitingForFinger();
void handleCollecting();
void handleTransmitting();
int sendStatus(const char* statusMsg);
void updateLcd(const char* statusMsg);
bool connectWiFi();
bool testServerConnection();
void clearCredentials();
void startConfigMode();

// ---------- HTML Page for Configuration ----------
const char* configPageHtml = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32 PPG Setup</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
    .container { max-width: 400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #333; text-align: center; font-size: 24px; }
    label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
    input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
    input[type="submit"] { width: 100%; padding: 12px; margin-top: 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
    input[type="submit"]:hover { background: #45a049; }
    .note { font-size: 12px; color: #888; margin-top: 5px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>ESP32 PPG Glucose Monitor</h1>
    <h2 style="text-align:center;color:#666;">WiFi Configuration</h2>
    <form action="/save" method="POST">
      <label>WiFi Network Name (SSID)</label>
      <input type="text" name="ssid" required placeholder="Enter WiFi name">
      
      <label>WiFi Password</label>
      <input type="password" name="password" required placeholder="Enter WiFi password">
      
      <label>Server URL</label>
      <input type="text" name="serverUrl" required placeholder="http://192.168.1.6:8000">
      <p class="note">For Cloud: https://your-service.run.app</p>
      
      <label>API Key</label>
      <input type="text" name="apiKey" required placeholder="Your API Key (e.g., XOZ35)">
      <p class="note">Get your API key from the web interface after login</p>
      
      <input type="submit" value="Save & Connect">
    </form>
  </div>
</body>
</html>
)rawliteral";

// ---------- Web Server Handlers ----------
void handleConfigRoot() {
  configServer.send(200, "text/html", configPageHtml);
}

void handleConfigSave() {
  if (configServer.hasArg("ssid") && configServer.hasArg("password") && configServer.hasArg("serverUrl")) {
    savedSSID = configServer.arg("ssid");
    savedPassword = configServer.arg("password");
    savedServerUrl = configServer.arg("serverUrl");
    savedApiKey = configServer.arg("apiKey");
    
    // Remove trailing slash from server URL if present
    if (savedServerUrl.endsWith("/")) {
      savedServerUrl = savedServerUrl.substring(0, savedServerUrl.length() - 1);
    }
    
    // Save to Preferences
    preferences.putString("ssid", savedSSID);
    preferences.putString("password", savedPassword);
    preferences.putString("serverUrl", savedServerUrl);
    preferences.putString("apiKey", savedApiKey);
    
    Serial.println("Configuration saved:");
    Serial.printf("  SSID: %s\n", savedSSID.c_str());
    Serial.printf("  Server: %s\n", savedServerUrl.c_str());
    Serial.printf("  API Key: %s\n", savedApiKey.c_str());
    
    // Send success response
    String response = "<html><body style='font-family:Arial;text-align:center;padding:50px;'>";
    response += "<h1>Configuration Saved!</h1>";
    response += "<p>Connecting to WiFi network: <b>" + savedSSID + "</b></p>";
    response += "<p>The device will restart and connect to your network.</p>";
    response += "</body></html>";
    configServer.send(200, "text/html", response);
    
    delay(2000);
    
    // Properly stop AP mode before transitioning
    Serial.println("Stopping AP mode...");
    configServer.stop();
    WiFi.softAPdisconnect(true);
    WiFi.mode(WIFI_OFF);  // Fully shutdown WiFi
    delay(500);  // Allow WiFi stack to clean up
    
    // Transition to connecting state
    currentState = CONNECTING_WIFI;
  } else {
    configServer.send(400, "text/plain", "Missing required fields");
  }
}

// ---------- WiFi Connection ----------
bool connectWiFi() {
  Serial.println("Connecting to WiFi...");
  Serial.printf("  SSID: %s\n", savedSSID.c_str());
  
  // Ensure WiFi is in correct mode with delay for stack initialization
  WiFi.mode(WIFI_STA);
  delay(100);  // Allow mode switch to complete
  
  WiFi.begin(savedSSID.c_str(), savedPassword.c_str());
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected!");
    Serial.print("  IP Address: ");
    Serial.println(WiFi.localIP());
    return true;
  } else {
    Serial.println("\n✗ Failed to connect to WiFi");
    return false;
  }
}

// ---------- Server Connection Test ----------
bool testServerConnection() {
  Serial.println("Testing server connection...");
  Serial.printf("  URL: %s\n", savedServerUrl.c_str());
  
  HTTPClient http;
  http.begin(savedServerUrl + "/");
  http.setTimeout(5000);
  int httpCode = http.GET();
  http.end();
  
  if (httpCode == 200) {
    Serial.println("✓ Server is reachable!");
    return true;
  } else {
    Serial.printf("✗ Server not reachable (HTTP %d)\n", httpCode);
    return false;
  }
}

// ---------- Clear Saved Credentials ----------
void clearCredentials() {
  Serial.println("Clearing saved credentials...");
  preferences.clear();
  savedSSID = "";
  savedPassword = "";
  savedServerUrl = "";
  savedApiKey = "";
}

// ---------- Start Configuration Mode ----------
void startConfigMode() {
  Serial.println("\n========================================");
  Serial.println("Starting WiFi Configuration Mode");
  Serial.println("========================================");
  
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(local_IP, local_IP, IPAddress(255, 255, 255, 0));
  WiFi.softAP(AP_SSID);
  
  Serial.printf("Connect to WiFi: %s\n", AP_SSID);
  Serial.printf("Then open: http://%s\n", WiFi.softAPIP().toString().c_str());
  
  // Setup web server routes
  configServer.on("/", handleConfigRoot);
  configServer.on("/save", HTTP_POST, handleConfigSave);
  configServer.begin();
  
  Serial.println("Web server started. Waiting for configuration...");
}

// ---------- Send Status Update ----------
int sendStatus(const char* statusMsg) {
  HTTPClient http;
  http.begin(statusUrl);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", savedApiKey);
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
  updateLcd(statusMsg);
  return httpCode;
}

void updateLcd(const char* statusMsg) {
  const char* modeLabel = (currentState == CONFIGURE_WIFI) ? "M0" : "M1";
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(modeLabel);
  lcd.print(" ");
  lcd.print(serverConnected ? "ServerOn" : "ServerOff");
  lcd.setCursor(0, 1);
  lcd.print(statusMsg);
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

  // Initialize Preferences
  preferences.begin("ppg-config", false);
  
  // Check for reset button held during boot
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);
  delay(100);  // Debounce
  if (digitalRead(RESET_BUTTON_PIN) == LOW) {
    Serial.println("\n*** BOOT button held - Clearing saved credentials ***");
    clearCredentials();
    delay(1000);
  }
  
  // Load saved credentials
  savedSSID = preferences.getString("ssid", "");
  savedApiKey = preferences.getString("apiKey", "");
  savedPassword = preferences.getString("password", "");
  savedServerUrl = preferences.getString("serverUrl", "");
  
  Serial.println("\nChecking saved configuration...");
  Serial.printf("  SSID: %s\n", savedSSID.length() > 0 ? savedSSID.c_str() : "(not set)");
  Serial.printf("  Server: %s\n", savedServerUrl.length() > 0 ? savedServerUrl.c_str() : "(not set)");
  Serial.printf("  API Key: %s\n", savedApiKey.length() > 0 ? savedApiKey.c_str() : "(not set)");
  
  // Determine initial state based on saved credentials
  if (savedSSID.length() == 0 || savedPassword.length() == 0 || savedServerUrl.length() == 0 || savedApiKey.length() == 0) {
    Serial.println("\nNo complete configuration found.");
    currentState = CONFIGURE_WIFI;
  } else {
    Serial.println("\nConfiguration found. Will attempt connection.");
    currentState = CONNECTING_WIFI;
  }

  lcd.init();
  lcd.backlight();
  updateLcd(currentState == CONFIGURE_WIFI ? "CONFIG" : "CONNECTING");

  // ECG lead-off inputs
  pinMode(LO_PLUS_PIN, INPUT_PULLUP);
  pinMode(LO_MINUS_PIN, INPUT_PULLUP);

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
  byte sampleAverage = 8;
  byte ledMode = 2;
  int sampleRate = 400;
  int pulseWidth = 411;
  int adcRange = 16384;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.clearFIFO();

  // Create esp_timer for precise sampling
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

  Serial.println("\nSetup complete.");
}

// ---------- MAIN LOOP ----------
void loop() {
  switch (currentState) {
    case CONFIGURE_WIFI:
      handleConfigureWiFi();
      break;

    case CONNECTING_WIFI:
      handleConnectingWiFi();
      break;

    case IDLE: {
      // Auto-transition to waiting for finger
      currentState = WAITING_FOR_FINGER;
      int statusCode = sendStatus("WAITING");
      
      // Check for authentication error (wrong API key)
      if (statusCode == 401) {
        Serial.println("\n*** Wrong API Key Entered ***");
        Serial.println("Returning to configuration mode...");
        // Clear only API key so user doesn't have to re-enter WiFi credentials
        preferences.remove("apiKey");
        savedApiKey = "";
        currentState = CONFIGURE_WIFI;
      }
      break;
    }

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

// ---------- STATE: Configure WiFi (AP Mode) ----------
void handleConfigureWiFi() {
  static bool apStarted = false;
  
  if (!apStarted) {
    startConfigMode();
    updateLcd("CONFIG");
    apStarted = true;
  }
  
  // Handle web server requests
  configServer.handleClient();
  
  // Check if we transitioned out of config mode
  if (currentState != CONFIGURE_WIFI) {
    apStarted = false;
  }
  
  delay(10);
}

// ---------- STATE: Connecting to WiFi ----------
void handleConnectingWiFi() {
  // Build endpoint URLs from saved server URL
  fullServerUrl = savedServerUrl + String(SERVER_ENDPOINT);
  statusUrl = savedServerUrl + "/status_update";
  completeUrl = savedServerUrl + "/collection_complete";
  
  Serial.println("\n========================================");
  Serial.println("Connecting to Network");
  Serial.println("========================================");
  updateLcd("CONNECTING");
  
  // Attempt WiFi connection
  if (!connectWiFi()) {
    Serial.println("WiFi connection failed. Returning to configuration mode.");
    serverConnected = false;
    updateLcd("RECONF");
    clearCredentials();
    currentState = CONFIGURE_WIFI;
    return;
  }
  
  // Test server connection
  delay(1000);
  if (!testServerConnection()) {
    Serial.println("Server connection failed. Returning to configuration mode.");
    serverConnected = false;
    updateLcd("RECONF");
    delay(1000);
    clearCredentials();
    currentState = CONFIGURE_WIFI;
    return;
  }
  serverConnected = true;
  updateLcd("READY");
  
  // Success! Transition to normal operation
  Serial.println("\n========================================");
  Serial.println("✓ Connected! Ready for operation.");
  Serial.println("========================================");
  Serial.println("Place finger on sensor to begin.");
  
  currentState = IDLE;
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
    uint16_t ecgValue = analogRead(ECG_PIN);
    
    // Quality tracking
    if (irValue < MIN_IR_THRESHOLD || irValue > MAX_IR_THRESHOLD) {
      poorQualitySamples++;
      Serial.printf("Poor quality sample detected: IR=%u\n", irValue);
      if (sampleTimer) {
        esp_timer_stop(sampleTimer);
      }
      sendStatus("Restarting due to poor quality");
      currentState = IDLE;
      return;
    }
    
    // Store sample (downscale from 18-bit to 16-bit by dividing by 4)
    ppgBuffer[sampleIndex] = (uint16_t)(irValue >> 2);
    ecgBuffer[sampleIndex] = ecgValue;
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
  
  auto sendStream = [&](uint16_t* buffer, const char* streamHeader) {
    for (int chunk = 0; chunk < totalChunks; chunk++) {
      int startIdx = chunk * SAMPLES_PER_CHUNK;
      int endIdx = min(startIdx + SAMPLES_PER_CHUNK, (int)TOTAL_SAMPLES);
      int chunkSamples = endIdx - startIdx;

      uint8_t packet[512];
      packet[0] = chunk & 0xFF;
      packet[1] = (chunk >> 8) & 0xFF;
      packet[2] = (uint8_t)totalChunks;
      packet[3] = (uint8_t)chunkSamples;

      memcpy(&packet[4], &buffer[startIdx], chunkSamples * sizeof(uint16_t));

      int packetSize = 4 + (chunkSamples * sizeof(uint16_t));

      bool sent = false;
      int retries = 3;

      for (int attempt = 0; attempt < retries && !sent; attempt++) {
        http.begin(fullServerUrl);
        http.addHeader("Content-Type", "application/octet-stream");
        http.addHeader("X-Chunk-Number", String(chunk));
        http.addHeader("X-Total-Chunks", String(totalChunks));
        http.addHeader("X-API-Key", savedApiKey);
        if (streamHeader && streamHeader[0] != '\0') {
          http.addHeader("X-Stream-Type", streamHeader);
        }
        http.setTimeout(15000);
        
        int httpCode = http.POST(packet, packetSize);
        
        if (httpCode == 200) {
          sent = true;
          float progress = (float)(chunk + 1) / totalChunks * 100.0;
          Serial.printf("%s Chunk %d/%d sent (%d samples, %.0f%%)\n",
                        streamHeader && streamHeader[0] ? streamHeader : "PPG",
                        chunk + 1, totalChunks, chunkSamples, progress);
        } else {
          Serial.printf("%s Chunk %d failed (attempt %d): HTTP %d\n", 
                        streamHeader && streamHeader[0] ? streamHeader : "PPG",
                        chunk + 1, attempt + 1, httpCode);
          if (attempt < retries - 1) {
            delay(1000);
          }
        }
        
        http.end();
      }
      
      if (!sent) {
        Serial.printf("ERROR: Failed to send %s chunk %d\n", streamHeader && streamHeader[0] ? streamHeader : "PPG", chunk + 1);
        currentState = IDLE;
        return false;
      }
      
      delay(50);
    }
    return true;
  };

  if (!sendStream(ppgBuffer, "PPG")) {
    return;
  }
  if (!sendStream(ecgBuffer, "ECG")) {
    return;
  }

  // Send collection_complete with metadata
  http.begin(completeUrl);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", savedApiKey);
  
  StaticJsonDocument<256> doc;
  doc["samples_collected"] = samplesCollected;
  doc["duration_seconds"] = collection_metadata_duration;
  doc["actual_sample_rate"] = collection_metadata_rate;
  doc["poor_quality_count"] = poorQualitySamples;
  doc["ecg_samples_collected"] = samplesCollected;
  doc["ecg_sample_rate"] = SAMPLING_RATE;
  doc["ecg_duration_seconds"] = collection_metadata_duration;
  
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
