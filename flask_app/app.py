from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from datetime import datetime
import time
import requests

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
 
# Telegram configuration
TELEGRAM_BOT_TOKEN = '8124143416:AAFF7mcmLiKeAhobDkcLIR89FzxeipbwIpY'
TELEGRAM_CHAT_ID = '734897504'
ALERT_INTERVAL = 10  # in seconds
last_alert_time = 0

# Location for alert (simple)
LOCATION = "Lab Entrance"

# Ensure folders exist
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Load MobileNet SSD model for person detection
# 👇👇👇 SET THE CORRECT PATH TO PROTOTXT AND CAFFEMODEL FILES HERE 👇👇👇
PROTOTXT_PATH = r'C:\Users\mithi\Downloads\face\deploy.prototxt'  # <- UPDATE PATH
MODEL_PATH = r'C:\Users\mithi\Downloads\face\mobilenet_iter_73000.caffemodel'   # <- UPDATE PATH

person_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
PERSON_CLASS_ID = 15  # 'person' class in MobileNetSSD

def send_telegram_alert(message, image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': message
        }
        files = {
            'photo': photo
        }
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
                return {
                    "detected": parts[1] == '1',
                    "timestamp": parts[0]
                }
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

    return render_template('index.html',
                           time=int(time.time()),
                           person_detected=person_detected,
                           timestamp=timestamp)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_data = request.data
        if not img_data:
            return 'No image data received', 400

        # Decode image
        npimg = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return 'Failed to decode image', 400

        # Person detection using MobileNet SSD
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        person_net.setInput(blob)
        detections = person_net.forward()

        person_detected = False
        (h, w) = frame.shape[:2]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if class_id == PERSON_CLASS_ID and confidence > 0.5:
                person_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save latest image
        save_path = os.path.join(app.config['STATIC_FOLDER'], 'latest.jpg')
        cv2.imwrite(save_path, frame)

        # Save detection status
        with open(os.path.join(app.config['STATIC_FOLDER'], 'meta.txt'), 'w') as meta_file:
            meta_file.write(f"{timestamp_str},{int(person_detected)}")

        # Send Telegram alert every 10 sec max
        global last_alert_time
        current_time = time.time()
        if person_detected and (current_time - last_alert_time > ALERT_INTERVAL):
            last_alert_time = current_time
            telegram_message = f"👤 Person detected at {timestamp_str} in {LOCATION}"
            send_telegram_alert(telegram_message, save_path)

        return 'OK', 200

    except Exception as e:
        print("❌ Error processing upload:", e)
        return 'Server error', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
