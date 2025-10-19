/*
 * PPG Data Collector for Blood Glucose Prediction
 * ESP32 + MAX30102 Sensor
 *
 * Collects 1 minute of PPG data at 100 Hz and transmits via BLE to app
 *
 * Hardware:
 *   - ESP32 DevKit (ESP32-WROOM-32)
 *   - MAX30102 Heart Rate & Pulse Oximeter Sensor
 *
 * Wiring:
 *   MAX30102 VIN  -> ESP32 3.3V
 *   MAX30102 GND  -> ESP32 GND
 *   MAX30102 SDA  -> ESP32 GPIO21 (I2C SDA)
 *   MAX30102 SCL  -> ESP32 GPIO22 (I2C SCL)
 *
 * Libraries Required:
 *   - SparkFun MAX3010x Pulse and Proximity Sensor Library (MAX30105.h)
 *   - BLE (built-in ESP32)
 */

#include <Wire.h>
#include "MAX30105.h"
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ---------- CONFIGURATION ----------
#define I2C_SDA 21  // Standard ESP32 SDA pin
#define I2C_SCL 22  // Standard ESP32 SCL pin

#define SAMPLING_RATE 100                 // Hz (samples per second)
#define COLLECTION_TIME 60                // seconds
#define TOTAL_SAMPLES (SAMPLING_RATE * COLLECTION_TIME)  // 6000 samples
#define SAMPLE_INTERVAL_MS (1000 / SAMPLING_RATE)        // 10ms for 100Hz

// BLE Configuration
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_DATA_UUID      "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHAR_STATUS_UUID    "beb5483e-36e1-4688-b7f5-ea07361b26a9"
#define CHAR_CONTROL_UUID   "beb5483e-36e1-4688-b7f5-ea07361b26aa"

// Quality thresholds
#define MIN_IR_THRESHOLD 50000     // Minimum IR for finger detection
#define MAX_IR_THRESHOLD 200000    // Maximum IR (saturation check)

// ---------- GLOBALS ----------
MAX30105 particleSensor;

// BLE
BLEServer* pServer = NULL;
BLECharacteristic* pDataCharacteristic = NULL;
BLECharacteristic* pStatusCharacteristic = NULL;
BLECharacteristic* pControlCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

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

// ---------- FORWARD DECLARATIONS ----------
void initBLE();
void handleWaitingForFinger();
void handleCollecting();
void handleTransmitting();
void sendStatus(const char* status);
void printSensorDiagnostics();

// ---------- BLE CALLBACKS ----------
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("BLE: Device connected");
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("BLE: Device disconnected");
  }
};

class ControlCharacteristicCallbacks: public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pCharacteristic) override {
    String value = pCharacteristic->getValue().c_str();
    if (value.length() > 0) {
      char command = value[0];
      switch (command) {
        case 'S':  // Start collection
          if (currentState == IDLE) {
            Serial.println("Command: START collection");
            currentState = WAITING_FOR_FINGER;
            sampleIndex = 0;
            samplesCollected = 0;
            poorQualitySamples = 0;
          }
          break;
        case 'C':  // Cancel
          Serial.println("Command: CANCEL");
          currentState = IDLE;
          sampleIndex = 0;
          break;
        case 'R':  // Reset
          Serial.println("Command: RESET");
          currentState = IDLE;
          sampleIndex = 0;
          samplesCollected = 0;
          break;
      }
    }
  }
};

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  Serial.println("PPG Data Collector - ESP32 + MAX30102");
  Serial.println("======================================");

  // Initialize I2C (explicit pins + fast clock)
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000); // 400 kHz I2C

  // Initialize MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found!");
    while (1) { delay(1000); }
  }
  Serial.println("MAX30102 initialized");

  // Configure MAX30102 for 100 Hz sampling
  byte ledBrightness = 0x1F;  // 6.4mA (0x00..0xFF)
  byte sampleAverage = 4;     // Average 4 samples
  byte ledMode = 2;           // Red + IR mode
  int sampleRate = 100;       // 100 sps
  int pulseWidth = 411;       // 411 µs
  int adcRange = 16384;       // 14-bit ADC

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x0A);  // Red LED current
  particleSensor.setPulseAmplitudeIR(0x1F);   // IR LED current (higher for better SNR)

  // Clear FIFO
  particleSensor.clearFIFO();

  Serial.println("Sensor configured for 100 Hz");

  // Initialize BLE
  initBLE();

  Serial.println("\nReady! Waiting for iOS app connection...");
  Serial.println("State: IDLE");
}

void initBLE() {
  // Create BLE Device
  BLEDevice::init("ESP32-PPG-Glucose");

  // Hint a reasonable MTU (iOS typically negotiates ~185)
  BLEDevice::setMTU(185);

  // Create BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Data Characteristic (PPG data -> notifications)
  pDataCharacteristic = pService->createCharacteristic(
    CHAR_DATA_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pDataCharacteristic->addDescriptor(new BLE2902());

  // Status Characteristic (status -> notify/read)
  pStatusCharacteristic = pService->createCharacteristic(
    CHAR_STATUS_UUID,
    BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ
  );
  pStatusCharacteristic->addDescriptor(new BLE2902());

  // Control Characteristic (commands from iOS)
  pControlCharacteristic = pService->createCharacteristic(
    CHAR_CONTROL_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  pControlCharacteristic->setCallbacks(new ControlCharacteristicCallbacks());

  // Start service
  pService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x06); // optional: helps with iOS connections
  BLEDevice::startAdvertising();

  Serial.println("BLE advertising started");
}

// ---------- MAIN LOOP ----------
void loop() {
  // Handle BLE connection changes
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    Serial.println("BLE: Restart advertising");
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }

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
    // Finger detected with good contact
    Serial.println("Finger detected! Starting collection...");
    sendStatus("COLLECTING");
    currentState = COLLECTING;
    collectionStartTime = millis();
    sampleIndex = 0;
    samplesCollected = 0;
    poorQualitySamples = 0;
    lastSampleTime = millis();
  } else {
    // Send periodic status updates
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

  // Sample at exact 100 Hz (every 10ms)
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    // Read IR value (more stable than Red for glucose estimation)
    uint32_t irValue = particleSensor.getIR();

    // Check signal quality
    bool goodQuality = (irValue > MIN_IR_THRESHOLD && irValue < MAX_IR_THRESHOLD);
    if (!goodQuality) {
      poorQualitySamples++;

      // If too many poor quality samples, abort
      if (poorQualitySamples > 100) {  // >1 second of poor quality
        Serial.println("ERROR: Poor signal quality - collection aborted");
        sendStatus("ERROR_QUALITY");
        currentState = IDLE;
        return;
      }
    }

    // Store sample (scale 32-bit to 16-bit)
    ppgBuffer[sampleIndex] = (uint16_t)(irValue >> 2);  // Divide by 4 to fit 16-bit
    sampleIndex++;
    samplesCollected++;

    // Progress updates every 10 samples (0.1 seconds)
    if (sampleIndex % 10 == 0) {
      float progress = (float)sampleIndex / TOTAL_SAMPLES * 100.0;
      char statusMsg[32];
      sprintf(statusMsg, "PROGRESS:%.1f", progress);
      sendStatus(statusMsg);

      // Serial progress
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
  if (!deviceConnected) {
    Serial.println("ERROR: Device disconnected during transmission");
    currentState = IDLE;
    return;
  }

  Serial.println("Starting BLE transmission...");

  // Transmit data in chunks with header.
  // MTU ~185 -> safe payload ~182 bytes.
  // Header: chunk_num(2) + total_chunks(1) + sample_count(1) = 4 bytes
  // Remaining: 178 bytes for samples (89 uint16_t samples)
  const int CHUNK_SIZE = 89;  // samples per chunk
  int totalChunks = (TOTAL_SAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE;

  for (int chunk = 0; chunk < totalChunks; chunk++) {
    int startIdx = chunk * CHUNK_SIZE;
    int endIdx = min(startIdx + CHUNK_SIZE, (int)TOTAL_SAMPLES);
    int chunkSamples = endIdx - startIdx;

    // Build packet with header: [chunk_num(2), total_chunks(1), sample_count(1), samples...]
    uint8_t packet[512];  // Large enough for header + samples

    // Header (4 bytes)
    packet[0] = chunk & 0xFF;              // chunk_num low byte
    packet[1] = (chunk >> 8) & 0xFF;       // chunk_num high byte
    packet[2] = (uint8_t)totalChunks;      // total_chunks (1 byte)
    packet[3] = (uint8_t)chunkSamples;     // sample_count (1 byte)

    // Copy samples (little-endian uint16_t)
    memcpy(&packet[4], &ppgBuffer[startIdx], chunkSamples * sizeof(uint16_t));

    // Send packet
    int packetSize = 4 + (chunkSamples * sizeof(uint16_t));
    pDataCharacteristic->setValue(packet, packetSize);
    pDataCharacteristic->notify();

    // Progress update
    float progress = (float)(chunk + 1) / totalChunks * 100.0;
    char statusMsg[32];
    sprintf(statusMsg, "TRANSMIT:%.1f", progress);
    sendStatus(statusMsg);

    Serial.printf("Transmitted chunk %d/%d: %d samples, %d bytes (%.1f%%)\n",
                  chunk + 1, totalChunks, chunkSamples, packetSize, progress);

    // Small delay to avoid overwhelming BLE stack
    delay(20);
  }

  Serial.println("Transmission complete!");
  sendStatus("COMPLETE");
  currentState = COMPLETE;
}

// ---------- HELPERS ----------
void sendStatus(const char* status) {
  if (deviceConnected && pStatusCharacteristic) {
    pStatusCharacteristic->setValue(status);
    pStatusCharacteristic->notify();
  }
}

// Optional diagnostics
void printSensorDiagnostics() {
  Serial.println("\n=== MAX30102 Diagnostics ===");
  uint32_t irValue = particleSensor.getIR();
  uint32_t redValue = particleSensor.getRed();
  Serial.printf("IR Value: %d\n", irValue);
  Serial.printf("Red Value: %d\n", redValue);

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
