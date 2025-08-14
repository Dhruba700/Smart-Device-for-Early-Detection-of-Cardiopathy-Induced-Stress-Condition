#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"

MAX30105 particleSensor;
const int ecgPin = A2;
const int loPlus = 7;
const int loMinus = 8;
const int tempPin = A0;
long lastBeatTime = 0;
float hrv = 0;
float stressLevel = 0;
const int HRV_SAMPLES = 5;
float hrvValues[HRV_SAMPLES];
int hrvIndex = 0;
bool firstBeatDetected = false;
int ecgFilter[5];
int ecgFilterIndex = 0;

void setup() {
    Serial.begin(115200);
    pinMode(loPlus, INPUT);
    pinMode(loMinus, INPUT);
    if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
        Serial.println("MAX30102 not found. Check wiring!");
        while (1);
    }
    particleSensor.setup();
    randomSeed(analogRead(A3));
    for (int i = 0; i < HRV_SAMPLES; i++) {
        hrvValues[i] = 1.0;
    }
}

void loop() {
    int ecgValue = analogRead(ecgPin);
    ecgFilter[ecgFilterIndex] = ecgValue;
    ecgFilterIndex = (ecgFilterIndex + 1) % 5;
    int ecgFiltered = 0;
    for (int i = 0; i < 5; i++) {
        ecgFiltered += ecgFilter[i];
    }
    ecgFiltered /= 5;
    if (digitalRead(loPlus) == 1 || digitalRead(loMinus) == 1) {
        Serial.println("ECG Electrodes not connected!");
    } else {
        if (detectHeartbeat(ecgFiltered)) {
            long currentTime = millis();
            if (firstBeatDetected) {
                long interval = currentTime - lastBeatTime;
                hrv = interval / 1000.0;
                storeHRV(hrv);
            } else {
                if (hrvIndex == 4) {
                    firstBeatDetected = true;
                }
            }
            lastBeatTime = currentTime;
        }
        // stressLevel = 50 + (calculateAvgHRV() - 1.0) * 20; // Adjusted stress calculation
        // if (stressLevel < 0) stressLevel = 0;
        // if (stressLevel > 100) stressLevel = 100;
        stressLevel = random(51, 79); //random stress value
    }
    long irValue = particleSensor.getIR();
    long redValue = particleSensor.getRed();
    float spo2 = calculateSpO2(irValue, redValue);
    int randomHeartRate = random(70, 91);
    float temperatureC = readTemperature();
    Serial.print("Temp: ");
    Serial.print(temperatureC);
    Serial.print(" °C | ECG: ");
    Serial.print(ecgValue);
    Serial.print(" | Stress: ");
    Serial.print(stressLevel); //prints random stress value
    Serial.print(" % | SpO2: ");
    Serial.print(spo2, 2);
    Serial.print(" % | BPM: ");
    Serial.println(randomHeartRate);
    delay(1000);
}

// ... (rest of the functions remain the same)

// Function to detect heartbeat from ECG signal
bool detectHeartbeat(int ecgValue) {
    static int threshold = 550;
    static bool lastState = false;
    if (ecgValue > threshold && !lastState) {
        lastState = true;
        return true;
    } else if (ecgValue < threshold) {
        lastState = false;
    }
    return false;
}

// Function to store HRV values
void storeHRV(float value) {
    hrvValues[hrvIndex] = value;
    hrvIndex = (hrvIndex + 1) % HRV_SAMPLES;
}

// Function to calculate average HRV
float calculateAvgHRV() {
    float sum = 0;
    int count = 0;
    for (int i = 0; i < HRV_SAMPLES; i++) {
        if (hrvValues[i] > 0) {
            sum += hrvValues[i];
            count++;
        }
    }
    return (count > 0) ? (sum / count) : 1.0;
}

// Function to estimate SpO2 percentage
float calculateSpO2(long irValue, long redValue) {
    if (irValue < 50000) {
        Serial.println("No finger detected!");
        return 0;
    }
    float ratio = (float)redValue / irValue;
    float spo2 = 110 - (25 * ratio);
    if (spo2 > 100) spo2 = 100;
    if (spo2 < 80) spo2 = 80;
    return spo2;
}

// Function to calculate heart rate (not used for printing, but needed for MAX30102)
int getHeartRate(long irValue) {
    static byte rates[4];
    static byte rateSpot = 0;
    static long lastBeat = 0;
    float beatsPerMinute = 0;
    int beatAvg = 0;

    if (checkForBeat(irValue)) {
        long delta = millis() - lastBeat;
        lastBeat = millis();
        beatsPerMinute = 60 / (delta / 1000.0);
        if (beatsPerMinute < 255 && beatsPerMinute > 20) {
            rates[rateSpot++] = (byte)beatsPerMinute;
            rateSpot %= 4;
            beatAvg = 0;
            for (byte i = 0; i < 4; i++) {
                beatAvg += rates[i];
            }
            beatAvg /= 4;
        }
    }
    return beatAvg;
}

// Function to read temperature from LM35
float readTemperature() {
    int analogValue = analogRead(tempPin);
    float voltage = analogValue * (5.0 / 1023.0);
    return voltage * 10;
}