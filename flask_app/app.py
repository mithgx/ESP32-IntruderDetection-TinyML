from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import requests
import cv2

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

# Telegram Config
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'XYZ'
ALERT_INTERVAL = 10  # seconds
last_alert_time = 0
LOCATION = "Lab Entrance"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ensure folders exist
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def detect_face_with_tflite(frame):
    resized = cv2.resize(frame, (96, 96))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    input_data = np.expand_dims(gray, axis=(0, -1)).astype(np.float32) / 255.0  # (1,96,96,1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])  # e.g., [[0.91]]
    return output[0][0] > 0.5

def send_telegram_alert(message, image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message}
        files = {'photo': photo}
        try:
            r = requests.post(url, data=data, files=files)
            if r.status_code == 200:
                print("✅ Telegram alert sent")
            else:
                print(f"❌ Telegram failed: {r.status_code}, {r.text}")
        except Exception as e:
            print("❌ Telegram Exception:", e)

@app.route('/status')
def status():
    meta_path = os.path.join(app.static_folder, 'meta.txt')
    if not os.path.exists(meta_path):
        return {"detected": False, "timestamp": None}
    with open(meta_path, 'r') as f:
        content = f.read().strip()
        if content:
            parts = content.split(',')
            if len(parts) == 2:
                return {"detected": parts[1] == '1', "timestamp": parts[0]}
    return {"detected": False, "timestamp": None}

@app.route('/')
def index():
    meta_path = os.path.join(app.config['STATIC_FOLDER'], 'meta.txt')
    person_detected = False
    timestamp = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            content = f.read().strip()
            if content:
                parts = content.split(',')
                if len(parts) == 2:
                    timestamp = parts[0]
                    person_detected = parts[1] == '1'
    return render_template('index.html', time=int(time.time()),
                           person_detected=person_detected,
                           timestamp=timestamp)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_data = request.data
        if not img_data:
            return 'No image data received', 400

        npimg = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return 'Failed to decode image', 400

        face_detected = detect_face_with_tflite(frame)
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_path = os.path.join(app.config['STATIC_FOLDER'], 'latest.jpg')
        cv2.imwrite(save_path, frame)

        with open(os.path.join(app.config['STATIC_FOLDER'], 'meta.txt'), 'w') as f:
            f.write(f"{timestamp_str},{int(face_detected)}")

        global last_alert_time
        current_time = time.time()
        if face_detected and (current_time - last_alert_time > ALERT_INTERVAL):
            last_alert_time = current_time
            msg = f"👤 Face detected at {timestamp_str} in {LOCATION}"
            send_telegram_alert(msg, save_path)

        return 'OK', 200

    except Exception as e:
        print("❌ Error processing upload:", e)
        return 'Server error', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
