# 🤖 ESP32-CAM Face Detection using TinyML + Flask Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success.svg" />

  <img src="https://img.shields.io/badge/TinyML-TFLite-green.svg" />
  <img src="https://img.shields.io/badge/python-3.x-blue.svg" />
</p>

## 🔍 Overview

This project demonstrates a **real-time face detection system** using:
- 📷 **ESP32-CAM (without PSRAM)** for capturing images
- 🧠 **TinyML model (.tflite)** running on a Python Flask backend
- 🌐 A web dashboard for displaying real-time detection and history
- 📲 **Telegram bot integration** for sending face detection alerts with image, timestamp, and location

Designed for **low-resource hardware**, this solution is lightweight and ideal for entry-level IoT AI applications.

---

## 🧰 Components Used

- ESP32-CAM (AI Thinker)
- TensorFlow Lite (TFLite) for model inference
- Flask Web Framework (Python)
- OpenCV for image handling
- Telegram Bot API for notifications

---

---

## 🚀 Features

- 📸 Real-time image capture from ESP32-CAM
- 🧠 Face detection using quantized TFLite model
- 💻 Web dashboard to view latest image + status
- 🔔 Instant Telegram alerts with location + timestamp
- 🧪 Training notebook for building your own model
- ⚡ Fast, works with low-resolution images (160x120)

---

## 🧑‍💻 Setup Guide

### 🔧 1. Flash ESP32-CAM

- Open `esp32/esp32_cam_sender.ino` in Arduino IDE
- Update WiFi credentials and Flask server IP:

````markdown

const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_PASSWORD";
const char* serverUrl = "http://<your-pc-ip>:8000/upload";

````

- Set board to **AI Thinker ESP32-CAM** and flash the code

### 🐍 2. Set Up Python Server

cd flask_app
pip install -r requirements.txt
python app.py


> Make sure `face_detection.tflite` is in `flask_app/model/`

Visit `http://localhost:8000` to open the dashboard.

---

## 📲 Telegram Alerts

### How to Enable

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Get your bot token and your own chat ID
3. Replace in `app.py`:

TELEGRAM_BOT_TOKEN = 'your_bot_token'
TELEGRAM_CHAT_ID = 'your_chat_id'
LOCATION = "Lab Entrance" # Customizable tag


---

## 🧠 Train Your Own Face Detection Model

- Open `notebook/train_and_convert.ipynb`
- Use a dataset like WIDER FACE or your custom face images
- Build a lightweight CNN model
- Export and quantize to TensorFlow Lite:

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()


- Save as `face_detection.tflite` and move it to `flask_app/model/`

---

## 💻 Web Dashboard

Dashboard (`/`) shows:
- 📷 Latest image with face bounding boxes
- ✅ Detection status (face/no-face)
- 🕒 Last detected time
- 🔄 Auto-refreshing UI

---

## 🧪 Testing

1. Power up ESP32-CAM
2. Server receives `POST` image every ~100ms
3. Flask runs inference on image using TFLite
4. If face is found:
   - ✅ Detection shown on dashboard
   - 📩 Telegram alert sent with image

---

## 📦 Requirements :
```
Flask==2.3.2
opencv-python==4.8.0.74
numpy==1.24.3
requests==2.31.0
tensorflow==2.12.0
```
```
