#include <DFRobot_MAX30102.h>

DFRobot_MAX30102 particleSensor;

// Moving average for baseline removal
#define BASELINE_WINDOW 50
float baselineSum = 0;
float baselineBuffer[BASELINE_WINDOW];
int baselineIndex = 0;

// Simple notch filter coefficients (50Hz at 300Hz sample rate)
#define NOTCH_COEFF_B0  0.9331
#define NOTCH_COEFF_B1 -1.8126
#define NOTCH_COEFF_B2  0.9331
#define NOTCH_COEFF_A1 -1.8126
#define NOTCH_COEFF_A2  0.8662

// Filter state variables (for 50Hz)
float notch50_x1 = 0, notch50_x2 = 0;
float notch50_y1 = 0, notch50_y2 = 0;

// Filter state variables (for 60Hz)
#define NOTCH60_COEFF_B0  0.9162
#define NOTCH60_COEFF_B1 -1.7140
#define NOTCH60_COEFF_B2  0.9162
#define NOTCH60_COEFF_A1 -1.7140
#define NOTCH60_COEFF_A2  0.8325

float notch60_x1 = 0, notch60_x2 = 0;
float notch60_y1 = 0, notch60_y2 = 0;

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 10; // 100 Hz for stable serial output

// Simple baseline removal using moving average
float removeBaseline(float input) {
  baselineSum -= baselineBuffer[baselineIndex];
  baselineBuffer[baselineIndex] = input;
  baselineSum += input;
  
  baselineIndex = (baselineIndex + 1) % BASELINE_WINDOW;
  
  float baseline = baselineSum / BASELINE_WINDOW;
  return input - baseline;
}

// Apply 50Hz notch filter
float notchFilter50Hz(float input) {
  float output = NOTCH_COEFF_B0 * input + 
                 NOTCH_COEFF_B1 * notch50_x1 + 
                 NOTCH_COEFF_B2 * notch50_x2 -
                 NOTCH_COEFF_A1 * notch50_y1 - 
                 NOTCH_COEFF_A2 * notch50_y2;
  
  notch50_x2 = notch50_x1;
  notch50_x1 = input;
  notch50_y2 = notch50_y1;
  notch50_y1 = output;
  
  return output;
}

// Apply 60Hz notch filter
float notchFilter60Hz(float input) {
  float output = NOTCH60_COEFF_B0 * input + 
                 NOTCH60_COEFF_B1 * notch60_x1 + 
                 NOTCH60_COEFF_B2 * notch60_x2 -
                 NOTCH60_COEFF_A1 * notch60_y1 - 
                 NOTCH60_COEFF_A2 * notch60_y2;
  
  notch60_x2 = notch60_x1;
  notch60_x1 = input;
  notch60_y2 = notch60_y1;
  notch60_y1 = output;
  
  return output;
}

void setup() {
  // Use standard baud rate for reliability
  Serial.begin(115200);
  Wire.begin(25, 33);
  Wire.setClock(400000);
  
  // Initialize baseline buffer
  for(int i = 0; i < BASELINE_WINDOW; i++) {
    baselineBuffer[i] = 0;
  }
  
  // Wait for sensor
  while (!particleSensor.begin()) {
    Serial.println("MAX30102 not found");
    delay(1000);
  }

  particleSensor.sensorConfiguration(
    0x1F,           // LED brightness
    SAMPLEAVG_4,    // Average 4 samples for stability
    MODE_MULTILED,  // Red + IR
    SAMPLERATE_400, // 400 Hz hardware sampling
    PULSEWIDTH_411, // Pulse width
    ADCRANGE_4096   // Range
  );
  
  delay(2000);
  
  Serial.println("Starting...");
  delay(100);
  
  lastSampleTime = millis();
}

void loop() {
  unsigned long currentTime = millis();
  
  // Sample at controlled rate (100 Hz = 10ms interval)
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    
    // Read raw IR value
    uint32_t rawIR = particleSensor.getIR();
    
    // Convert to float for processing
    float rawFloat = (float)rawIR;
    
    // Apply processing pipeline
    float noBaseline = removeBaseline(rawFloat);
    float no50Hz = notchFilter50Hz(noBaseline);
    float processed = notchFilter60Hz(no50Hz);
    
    // Output for Serial Plotter
    // Format: value1 value2 value3 value4 (space-separated for plotter)
    Serial.print(rawFloat);
    Serial.print(" ");
    Serial.print(noBaseline);
    Serial.print(" ");
    Serial.print(no50Hz);
    Serial.print(" ");
    Serial.println(processed);
    
    lastSampleTime = currentTime;
  }
}