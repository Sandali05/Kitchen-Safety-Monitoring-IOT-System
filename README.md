# 🔥 Smart Kitchen Safety Monitoring System (IoT + TinyML)

An intelligent **IoT-based kitchen safety system** powered by **ESP32, TinyML, and Firebase**.  
This project monitors real-time environmental conditions and detects potential risks such as **gas leaks, fire, high temperature, and unsafe conditions**, triggering alerts and automated actions.

---

## 🚀 Features

- 📡 Real-time sensor monitoring (Temperature, Humidity, Gas, Flame, Fridge Door)
- 🧠 TinyML-based risk classification (SAFE / WARNING / DANGER)
- ⚠️ Anomaly detection using statistical thresholds
- 🌐 Firebase Realtime Database integration
- 🔔 Smart alerts (Buzzer, LEDs)
- 🌬 Automatic fan control for high temperature
- 📊 Cloud-based logging & historical data tracking
- 🔄 Tree-based fallback model for reliability
- 🛡 Safety override logic for critical situations

---

## 🧰 Hardware Components

- ESP32
- DHT22 Temperature & Humidity Sensor
- MQ Gas Sensor
- Flame Sensor (Analog + Digital)
- Magnetic Door Sensor (Fridge)
- Buzzer
- LEDs (Green, Yellow, Red)
- Relay Module (Fan)

---

## 🧠 AI Model

- TensorFlow Lite (TinyML)
- Quantized model (`model.h`)
- Feature normalization (`model_config.h`)
- Anomaly detection (`anomaly_config.h`)

### Risk Classes:
- ✅ SAFE
- ⚠️ WARNING
- 🔥 DANGER

---

## 🔌 System Architecture

Sensors → ESP32 → TinyML Inference → Decision Logic → Actuators + Firebase

---

## 📡 Firebase Structure

```
TempnHumData/
gassensorData/
FridgeDoor/
fire_monitoring/
```

Each includes:
- `latest`
- `history`
- `analysis`

---

## ⚙️ Setup Instructions

### Install Required Libraries

- WiFi.h
- Firebase ESP Client
- DHT Sensor Library
- TensorFlowLite_ESP32

---

## ⚡ System Behavior

| Condition | Action |
|----------|--------|
| High Temperature | Fan ON |
| Gas Leak | Buzzer ON |
| Fire Detected | Buzzer + RED LED |
| Warning State | Yellow LED |
| Safe State | Green LED |

---

## 📊 Data Uploaded to Firebase

- Sensor readings
- Risk predictions
- Anomaly detection
- Alert status
- Device status

---

## 🧪 AI + Rule Hybrid Logic

1. TinyML prediction
2. Confidence check
3. Tree-based fallback
4. Safety overrides (critical rules)

---

## 📌 Future Improvements

- Mobile app integration
- Push notifications
- Camera-based fire detection
- Edge dashboard UI (React)

---

## 📜 License

MIT License

---

## 👩‍💻 Author

Developed as part of an IoT + Edge AI project for smart kitchen safety.
