# 🤖 ESP32 Face Detection with TinyML + Flask Dashboard

This project showcases a real-time **face detection system** built using:
- 📷 **ESP32-CAM** (without PSRAM)
- 🧠 **TinyML (.tflite model)**
- 🌐 **Flask Web Dashboard**
- 📲 **Telegram Alerts**

The system is lightweight and works even on low-resolution images (e.g., 160x120). It is designed to **detect faces on the server side** after receiving images from the ESP32 and notify the user if a face is found.

---

## 🚀 Features

- 🔁 **Real-time image streaming** from ESP32-CAM via HTTP
- 🧠 **Face detection using TensorFlow Lite**
- 🖼️ **Web dashboard** to view latest image & detection logs
- 🔔 **Telegram alerts** with image, timestamp, and location
- 📒 Jupyter notebook to **train and convert models**
- ✅ Optimized for **low-resolution 160x120 images**

---

## 📁 Folder Structure


