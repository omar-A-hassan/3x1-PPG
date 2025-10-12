#include <DFRobot_MAX30102.h>

DFRobot_MAX30102 particleSensor;

unsigned long startTime;
unsigned long sampleCount = 0;
unsigned long lastSampleTime = 0;

// Target sample rate: 300 Hz = 3.333ms per sample
const unsigned long SAMPLE_INTERVAL_US = 3333; // microseconds

void setup()
{
  Serial.begin(921600); // Higher baud rate for 300 Hz (was 115200)
  Wire.begin(25, 33);
  Wire.setClock(400000); // Set I2C to 400kHz (Fast Mode)
  
  while (!particleSensor.begin()) {
    Serial.println("MAX30102 was not found");
    delay(1000);
  }

  // Configure for maximum sample rate
  // SAMPLERATE_400 with SAMPLEAVG_1 gives ~400Hz, we'll pace it to 300Hz
  particleSensor.sensorConfiguration(
    /*ledBrightness=*/0x1F, 
    /*sampleAverage=*/SAMPLEAVG_1,  // No averaging for maximum speed
    /*ledMode=*/MODE_MULTILED, 
    /*sampleRate=*/SAMPLERATE_400,   // Hardware samples at 400Hz
    /*pulseWidth=*/PULSEWIDTH_411, 
    /*adcRange=*/ADCRANGE_4096
  );
  
  delay(1000);
  
  // Compact CSV header
  Serial.println("N,T,R,I"); // Sample Number, Timestamp(ms), Red, IR
  
  startTime = micros();
  lastSampleTime = startTime;
}

void loop()
{
  unsigned long currentTime = micros();
  
  // Check if it's time for next sample (non-blocking timing)
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_US) {
    
    // Read sensor data
    uint32_t redValue = particleSensor.getRed();
    uint32_t irValue = particleSensor.getIR();
    
    // Compact output format to reduce serial bandwidth
    // Using milliseconds instead of seconds for smaller numbers
    unsigned long timeMs = (currentTime - startTime) / 1000;
    
    Serial.print(sampleCount);
    Serial.print(",");
    Serial.print(timeMs);
    Serial.print(",");
    Serial.print(redValue);
    Serial.print(",");
    Serial.println(irValue);
    
    sampleCount++;
    lastSampleTime = currentTime;
  }
  
  // No delays! This loop runs as fast as possible
  // The timing is controlled by the if statement above
}
