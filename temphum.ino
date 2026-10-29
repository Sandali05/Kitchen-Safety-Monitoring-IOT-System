/*******************************************************
 * ESP32 Edge AI + Firebase IoT Project
 * 
 * Uses:
 * - TinyML model from model.h
 * - normalization from model_config.h
 * - anomaly threshold from anomaly_config.h
 * - tree fallback + safety override
 * - Firebase RTDB upload
 *******************************************************/

#include <WiFi.h>
#include <time.h>
#include <Firebase_ESP_Client.h>
#include <addons/TokenHelper.h>
#include <addons/RTDBHelper.h>
#include <DHT.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "model_config.h"
#include "anomaly_config.h"

// =====================================================
// WiFi + Firebase Config
// =====================================================
#define WIFI_SSID       "#"
#define WIFI_PASSWORD   "#"

#define API_KEY         ""
#define DATABASE_URL    ""
#define USER_EMAIL      ""
#define USER_PASSWORD   ""

// =====================================================
// Pins
// =====================================================
#define DHTPIN              19
#define DHTTYPE             DHT22

#define GAS_ANALOG_PIN      34
#define FLAME_ANALOG_PIN    35
#define FLAME_DIGITAL_PIN   4
#define FRIDGE_DOOR_PIN     14

#define BUZZER_PIN          26
#define GREEN_LED_PIN       18
#define YELLOW_LED_PIN      21
#define RED_LED_PIN         23
#define FAN_PIN             5

// =====================================================
// Hardware behavior switches
// =====================================================
// Fan relay modules are often active LOW
#define RELAY_ON  LOW
#define RELAY_OFF HIGH

// Flame sensor digital outputs are often active LOW.
// If yours is reversed, change LOW to HIGH below.
#define FLAME_DETECTED_LEVEL LOW

// If LEDs behave opposite, change LED_ON/LED_OFF.
#define LED_ON  HIGH
#define LED_OFF LOW

#define BUZZER_ON  LOW
#define BUZZER_OFF HIGH
// =====================================================
// Timing
// =====================================================
unsigned long lastUploadMillis = 0;
const unsigned long uploadIntervalMs = 5000;

// =====================================================
// DHT
// =====================================================
DHT dht(DHTPIN, DHTTYPE);

// =====================================================
// Firebase
// =====================================================
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// =====================================================
// TinyML
// =====================================================
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* tflModel = nullptr;
tflite::MicroMutableOpResolver<4> resolver;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// =====================================================
// Runtime
// =====================================================
unsigned long fridgeDoorOpenStart = 0;
float fridgeOpenDurationSec = 0.0;
int fireCount = 0;
bool lastDangerState = false;

String currentBuzzerStatus = "OFF";
String currentFireLedStatus = "OFF";
String currentFanStatus = "OFF";
String currentAlertLevel = "NONE";
bool currentAlertTriggered = false;

struct TinyMLResult {
  int predictedClass;
  float confidence;
  bool anomaly;
};

// =====================================================
// Time helpers
// =====================================================
void setupTime() {
  configTime(19800, 0, "pool.ntp.org", "time.nist.gov");
  Serial.print("Syncing time");
  time_t now = time(nullptr);
  int retry = 0;
  while (now < 100000 && retry < 30) {
    delay(500);
    Serial.print(".");
    now = time(nullptr);
    retry++;
  }
  Serial.println();
}

String getDateString() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "1970-01-01";
  char buffer[11];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d", &timeinfo);
  return String(buffer);
}

String getTimeString() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "00:00:00";
  char buffer[9];
  strftime(buffer, sizeof(buffer), "%H:%M:%S", &timeinfo);
  return String(buffer);
}

String getDateTimeString() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "1970-01-01 00:00:00";
  char buffer[20];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(buffer);
}

unsigned long getTimestamp() {
  time_t now = time(nullptr);
  if (now < 100000) return millis() / 1000;
  return (unsigned long)now;
}

// =====================================================
// WiFi
// =====================================================
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());
}

// =====================================================
// Firebase
// =====================================================
void setupFirebase() {
  Serial.println("Starting Firebase setup...");

  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback;

  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;

  Serial.println("About to call Firebase.begin()");
  Firebase.begin(&config, &auth);
  Serial.println("Firebase.begin() returned");

  Firebase.reconnectWiFi(true);

  Serial.println("Waiting for Firebase auth...");
  unsigned long start = millis();
  while (!Firebase.ready() && millis() - start < 20000) {
    Serial.print(".");
    delay(300);
  }
  Serial.println();

  if (Firebase.ready()) {
    Serial.println("Firebase initialized");
  } else {
    Serial.println("Firebase not ready");
    Serial.print("Error: ");
    Serial.println(fbdo.errorReason());
  }
}

// =====================================================
// TinyML setup
// =====================================================
void setupTinyML() {
  Serial.println("TinyML: loading model...");

  tflModel = tflite::GetModel(risk_classifier_model_tflite);
  if (tflModel == nullptr) {
    Serial.println("TinyML: model pointer null");
    while (1) delay(1000);
  }

  if (resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("TinyML: AddFullyConnected failed");
    while (1) delay(1000);
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("TinyML: AddSoftmax failed");
    while (1) delay(1000);
  }
  if (resolver.AddQuantize() != kTfLiteOk) {
    Serial.println("TinyML: AddQuantize failed");
    while (1) delay(1000);
  }
  if (resolver.AddDequantize() != kTfLiteOk) {
    Serial.println("TinyML: AddDequantize failed");
    while (1) delay(1000);
  }

  static tflite::MicroInterpreter static_interpreter(
    tflModel, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("TinyML: AllocateTensors failed");
    while (1) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("TinyML: input/output tensor null");
    while (1) delay(1000);
  }

  Serial.println("TinyML: model loaded successfully");
}

// =====================================================
// Sensor helpers
// =====================================================
String getFridgeDoorStatus() {
  int state = digitalRead(FRIDGE_DOOR_PIN);
  return (state == HIGH) ? "OPEN" : "CLOSED";
}

float getFridgeOpenDurationSec(String doorStatus) {
  if (doorStatus == "OPEN") {
    if (fridgeDoorOpenStart == 0) fridgeDoorOpenStart = millis();
    return (millis() - fridgeDoorOpenStart) / 1000.0f;
  } else {
    fridgeDoorOpenStart = 0;
    return 0.0f;
  }
}

String getTempStatus(float temperature) {
  if (temperature >= 40.0) return "DANGER";
  if (temperature > 33.0) return "WARNING";
  return "SAFE";
}

String getTempWarning(float temperature) {
  if (temperature >= 40.0) return "Danger! Critical Temperature Detected";
  if (temperature > 33.0) return "Warning! High Temperature Detected";
  return "Temperature Normal";
}

int readGasValue() {
  return analogRead(GAS_ANALOG_PIN);
}

int readFlameAnalog() {
  return analogRead(FLAME_ANALOG_PIN);
}

int readFlameDigital() {
  return digitalRead(FLAME_DIGITAL_PIN);
}

float mapFlameIntensityPercent(int raw) {
  float percent = ((4095.0f - raw) / 4095.0f) * 100.0f;
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  return percent;
}

// =====================================================
// TinyML helpers
// =====================================================
void normalizeFeatures(float raw[NUM_FEATURES], float normalized[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    if (FEATURE_STDS[i] != 0.0f) {
      normalized[i] = (raw[i] - FEATURE_MEANS[i]) / FEATURE_STDS[i];
    } else {
      normalized[i] = raw[i];
    }
  }
}

bool isAnomaly(float raw[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    float z = 0.0f;
    if (FEATURE_STDS[i] != 0.0f) {
      z = (raw[i] - FEATURE_MEANS[i]) / FEATURE_STDS[i];
    }
    if (z < 0) z = -z;
    if (z > ANOMALY_Z_THRESHOLD) return true;
  }
  return false;
}

int8_t quantizeInput(float value, float scale, int zero_point) {
  int32_t q = (int32_t)round(value / scale) + zero_point;
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

TinyMLResult runTinyML(float lp_gas_value, float temperature, float humidity, float flame_status_num, float fridge_open_duration_sec) {
  TinyMLResult result;
  result.predictedClass = 0;
  result.confidence = 0.0f;
  result.anomaly = false;

  if (interpreter == nullptr || input == nullptr || output == nullptr) {
    Serial.println("TinyML: not initialized");
    result.anomaly = true;
    return result;
  }

  float rawFeatures[NUM_FEATURES] = {
    lp_gas_value,
    temperature,
    humidity,
    flame_status_num,
    fridge_open_duration_sec
  };

  float normalized[NUM_FEATURES];
  normalizeFeatures(rawFeatures, normalized);

  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.int8[i] = quantizeInput(normalized[i], input_scale, input_zero_point);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("TinyML: Invoke failed");
    result.anomaly = true;
    return result;
  }

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  float probs[NUM_CLASSES];
  int bestIndex = 0;
  float bestProb = -9999.0f;

  for (int i = 0; i < NUM_CLASSES; i++) {
    probs[i] = (output->data.int8[i] - output_zero_point) * output_scale;
    if (probs[i] > bestProb) {
      bestProb = probs[i];
      bestIndex = i;
    }
  }

  result.predictedClass = bestIndex;
  result.confidence = bestProb;
  result.anomaly = isAnomaly(rawFeatures) || (bestProb < CONFIDENCE_THRESHOLD);

  return result;
}

// =====================================================
// Tree fallback
// =====================================================
int treeRiskClass(float temperature, float lp_gas_value, float fridge_open_duration_sec) {
  if (temperature <= 33.76) {
    if (lp_gas_value <= 448.48) {
      if (fridge_open_duration_sec <= 60.21) return 0;
      else return 1;
    } else {
      return 1;
    }
  } else {
    if (lp_gas_value <= 723.29) {
      if (lp_gas_value <= 353.28) {
        if (fridge_open_duration_sec <= 62.82) return 0;
        else return 1;
      } else {
        if (temperature <= 42.60) return 1;
        else return 2;
      }
    } else {
      if (temperature <= 41.32) return 1;
      else return 2;
    }
  }
}

int applySafetyOverrides(int baseClass, float temperature, int gasValue, String flameStatus) {
  int result = baseClass;

  if (flameStatus == "FIRE_DETECTED") result = 2;

  if (temperature >= 45.0) result = 2;
  else if (temperature >= 35.0 && result < 1) result = 1;

  if (gasValue >= 700) result = 2;
  else if (gasValue >= 400 && result < 1) result = 1;

  return result;
}

String classToLabel(int c) {
  if (c == 0) return "SAFE";
  if (c == 1) return "WARNING";
  return "DANGER";
}

// =====================================================
// Actuators
// =====================================================
void applyAlertOutputs(
  int riskClass,
  String flameStatus,
  int gasValue,
  float temperature
) {
  bool flameDetected = (flameStatus == "FIRE_DETECTED");
  bool gasLeak = (gasValue >= 400);
  bool highTemperature = (temperature >= 33.0);

  // LEDs show overall system risk
  if (riskClass == 0) {
    digitalWrite(GREEN_LED_PIN, LED_ON);
    digitalWrite(YELLOW_LED_PIN, LED_OFF);
    digitalWrite(RED_LED_PIN, LED_OFF);
    currentFireLedStatus = "GREEN";
    currentAlertLevel = "NONE";
    currentAlertTriggered = false;
  }
  else if (riskClass == 1) {
    digitalWrite(GREEN_LED_PIN, LED_OFF);
    digitalWrite(YELLOW_LED_PIN, LED_ON);
    digitalWrite(RED_LED_PIN, LED_OFF);
    currentFireLedStatus = "YELLOW";
    currentAlertLevel = "WARNING";
    currentAlertTriggered = true;
  }
  else {
    digitalWrite(GREEN_LED_PIN, LED_OFF);
    digitalWrite(YELLOW_LED_PIN, LED_OFF);
    digitalWrite(RED_LED_PIN, LED_ON);
    currentFireLedStatus = "RED";
    currentAlertLevel = "DANGER";
    currentAlertTriggered = true;
  }

  // Buzzer ONLY for flame or gas leak
  if (flameDetected || gasLeak) {
    digitalWrite(BUZZER_PIN, BUZZER_ON);
    currentBuzzerStatus = "ON";
  } else {
    digitalWrite(BUZZER_PIN, BUZZER_OFF);
    currentBuzzerStatus = "OFF";
  }

  // Fan relay ONLY for high temperature
  if (highTemperature) {
    digitalWrite(FAN_PIN, LOW);   // relay ON
    currentFanStatus = "ON";
  } else {
    digitalWrite(FAN_PIN, HIGH);  // relay OFF
    currentFanStatus = "OFF";
  }
}

// =====================================================
// Firebase writers
// =====================================================
bool setJSON(const String& path, FirebaseJson& json) {
  if (Firebase.RTDB.setJSON(&fbdo, path.c_str(), &json)) {
    return true;
  } else {
    Serial.print("Firebase setJSON failed: ");
    Serial.println(fbdo.errorReason());
    return false;
  }
}

void uploadTempHum(float temperature, float humidity, unsigned long ts) {
  String dateStr = getDateString();
  String timeStr = getTimeString();
  String status = getTempStatus(temperature);
  String warning = getTempWarning(temperature);

  FirebaseJson latestJson;
  latestJson.set("date", dateStr);
  latestJson.set("humidity", humidity);
  latestJson.set("status", status);
  latestJson.set("temperature", temperature);
  latestJson.set("time", timeStr);
  latestJson.set("timestamp", (int)ts);
  latestJson.set("warning", warning);

  FirebaseJson historyJson = latestJson;

  setJSON("/TempnHumData/latest", latestJson);
  setJSON("/TempnHumData/history/" + String(ts), historyJson);
}

void uploadFridgeDoor(String doorStatus, unsigned long ts) {
  FirebaseJson latestJson;
  latestJson.set("status", doorStatus);
  latestJson.set("Duration", fridgeOpenDurationSec);

  FirebaseJson historyJson;
  historyJson.set("status", doorStatus);
  historyJson.set("timestamp", (int)ts);
  historyJson.set("Duration", (float)fridgeOpenDurationSec);

  setJSON("/FridgeDoor/latest", latestJson);
  setJSON("/FridgeDoor/history/" + String(ts), historyJson);
}

void uploadFireMonitoring(
  String flameStatus,
  int aoRawValue,
  int doValue,
  int flameIntensity,
  int intensityPercent,
  unsigned long ts
) {
  FirebaseJson latestJson;
  latestJson.set("alert_level", currentAlertLevel);
  latestJson.set("alert_triggered", currentAlertTriggered);
  latestJson.set("ao_raw_value", aoRawValue);
  latestJson.set("buzzer_status", currentBuzzerStatus);
  latestJson.set("date_time", getDateTimeString());
  latestJson.set("do_value", doValue);
  latestJson.set("esp32_status", "ONLINE");
  latestJson.set("fire_count", fireCount);
  latestJson.set("fire_led_status", currentFireLedStatus);
  latestJson.set("flame_intensity", flameIntensity);
  latestJson.set("flame_status", flameStatus);
  latestJson.set("intensity_percent", intensityPercent);
  latestJson.set("fan_status", currentFanStatus);

  FirebaseJson historyJson = latestJson;

  setJSON("/fire_monitoring/latest", latestJson);
  setJSON("/fire_monitoring/history/" + String(ts), historyJson);
}

void uploadGasSensor(int gasValue, String status, unsigned long ts) {
  FirebaseJson latestJson;
  latestJson.set("gasValue", gasValue);
  latestJson.set("id", "gas_01");
  latestJson.set("status", status);
  latestJson.set("timestamp", (int)ts);

  FirebaseJson historyJson = latestJson;

  setJSON("/gassensorData/latest", latestJson);
  setJSON("/gassensorData/history/" + String(ts), historyJson);
}

void uploadAnalysis(
  TinyMLResult tinyResult,
  int treeClass,
  int finalClass,
  unsigned long ts,
  float fridge_open_duration_sec
) {
  String tinyLabel = classToLabel(tinyResult.predictedClass);
  String treeLabel = classToLabel(treeClass);
  String finalLabel = classToLabel(finalClass);

  FirebaseJson commonAnalysis;
  commonAnalysis.set("predicted_risk", finalLabel);
  commonAnalysis.set("tinyml_risk", tinyLabel);
  commonAnalysis.set("tree_risk", treeLabel);
  commonAnalysis.set("anomaly_detected", tinyResult.anomaly);
  commonAnalysis.set("confidence", tinyResult.confidence);
  commonAnalysis.set("updated_at", (int)ts);

  FirebaseJson tempAnalysis = commonAnalysis;
  setJSON("/TempnHumData/analysis", tempAnalysis);

  FirebaseJson fridgeAnalysis = commonAnalysis;
  fridgeAnalysis.set("door_open_duration_sec", fridge_open_duration_sec);
  setJSON("/FridgeDoor/analysis", fridgeAnalysis);

  FirebaseJson fireAnalysis = commonAnalysis;
  fireAnalysis.set("alert_level", currentAlertLevel);
  fireAnalysis.set("buzzer_status", currentBuzzerStatus);
  fireAnalysis.set("fire_led_status", currentFireLedStatus);
  fireAnalysis.set("fan_status", currentFanStatus);
  setJSON("/fire_monitoring/analysis", fireAnalysis);

  FirebaseJson gasAnalysis = commonAnalysis;
  setJSON("/gassensorData/analysis", gasAnalysis);
}

// =====================================================
// Setup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(FRIDGE_DOOR_PIN, INPUT_PULLUP);
  pinMode(FLAME_DIGITAL_PIN, INPUT_PULLUP);

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(YELLOW_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, BUZZER_OFF);
  digitalWrite(GREEN_LED_PIN, LED_OFF);
  digitalWrite(YELLOW_LED_PIN, LED_OFF);
  digitalWrite(RED_LED_PIN, LED_OFF);
  digitalWrite(FAN_PIN, RELAY_OFF);

  dht.begin();
  Serial.println("DHT begin done");

  connectWiFi();
  Serial.println("WiFi done");

  setupTime();
  Serial.println("Time done");
  
  setupFirebase();
  Serial.println("Firebase done");

  setupTinyML();
  Serial.println("TinyML done");
}

// =====================================================
// Loop
// =====================================================
void loop() {
  if (millis() - lastUploadMillis < uploadIntervalMs) return;
  lastUploadMillis = millis();

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read DHT sensor.");
    temperature = 0.0f;
    humidity = 0.0f;
  }

  int gasValue = readGasValue();
  int flameAnalog = readFlameAnalog();
  int flameDigital = readFlameDigital();
  int intensityPercent = (int)round(mapFlameIntensityPercent(flameAnalog));

  String doorStatus = getFridgeDoorStatus();
  fridgeOpenDurationSec = getFridgeOpenDurationSec(doorStatus);

  String flameStatus = (flameDigital == FLAME_DETECTED_LEVEL) ? "FIRE_DETECTED" : "SAFE";
  float flameStatusNum = (flameStatus == "SAFE") ? 0.0f : 1.0f;

  TinyMLResult tinyResult = runTinyML(
    (float)gasValue,
    temperature,
    humidity,
    flameStatusNum,
    fridgeOpenDurationSec
  );

  int treeClass = treeRiskClass(
    temperature,
    (float)gasValue,
    fridgeOpenDurationSec
  );

  int aiClass = tinyResult.predictedClass;
  if (tinyResult.confidence < CONFIDENCE_THRESHOLD) {
    aiClass = treeClass;
  }

  int finalClass = applySafetyOverrides(
    aiClass,
    temperature,
    gasValue,
    flameStatus
  );

  String finalRiskLabel = classToLabel(finalClass);

  bool isDangerNow = (finalClass == 2 || flameStatus == "FIRE_DETECTED");
  if (isDangerNow && !lastDangerState) fireCount++;
  lastDangerState = isDangerNow;

  applyAlertOutputs(finalClass, flameStatus, gasValue, temperature);

  unsigned long ts = getTimestamp();

  uploadTempHum(temperature, humidity, ts);
  uploadFridgeDoor(doorStatus, ts);
  uploadFireMonitoring(
    flameStatus,
    flameAnalog,
    flameDigital,
    flameAnalog,
    intensityPercent,
    ts
  );
  uploadGasSensor(gasValue, finalRiskLabel, ts);
  uploadAnalysis(
    tinyResult,
    treeClass,
    finalClass,
    ts,
    fridgeOpenDurationSec
  );

  Serial.println("================================");
  Serial.println("Sensor Readings:");
  Serial.print("Temperature: "); Serial.println(temperature);
  Serial.print("Humidity: "); Serial.println(humidity);
  Serial.print("Gas Value: "); Serial.println(gasValue);
  Serial.print("Fridge Door: "); Serial.println(doorStatus);
  Serial.print("Fridge Open Duration: "); Serial.println(fridgeOpenDurationSec);
  Serial.print("Flame Analog: "); Serial.println(flameAnalog);
  Serial.print("Flame Digital: "); Serial.println(flameDigital);
  Serial.print("Flame Status: "); Serial.println(flameStatus);
  Serial.print("Flame Intencity: "); Serial.println(intensityPercent);

  Serial.println("---- AI Results ----");
  Serial.print("TinyML Class: "); Serial.println(classToLabel(tinyResult.predictedClass));
  Serial.print("TinyML Confidence: "); Serial.println(tinyResult.confidence, 4);
  Serial.print("TinyML Anomaly: "); Serial.println(tinyResult.anomaly ? "YES" : "NO");
  Serial.print("Tree Class: "); Serial.println(classToLabel(treeClass));
  Serial.print("Final Class: "); Serial.println(finalRiskLabel);

  Serial.println("---- Actuators ----");
  Serial.print("Buzzer: "); Serial.println(currentBuzzerStatus);
  Serial.print("LED: "); Serial.println(currentFireLedStatus);
  Serial.print("Fan: "); Serial.println(currentFanStatus);
  Serial.println("================================");
  
}
