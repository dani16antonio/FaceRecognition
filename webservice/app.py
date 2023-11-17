import cv2
import base64
import os
import re
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

from utils.life_detector import LifeDetector
from utils.face_recognition import FaceRecognition

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
socketio = SocketIO(app)

life_detector = LifeDetector()
face_recognition = FaceRecognition()
blink_pattern = '01+0'
blink_detections = []

@app.route("/")
def authenticate():
    return render_template('index.html')


def detect_blink(frame):
    detection = life_detector.detect(frame)
    return detection


@socketio.on('image')
def receive_image(encoded_data):
    global blink_detections
    is_detected = False
    label = None
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = frame[:, :, [2,1,0] ]

    is_blink = detect_blink(frame)
    if not is_blink is None:
        blink_detections.append(int(is_blink>=0.95))
    if re.search(blink_pattern, ''.join(map(str, blink_detections))):
        label = face_recognition.recognize(frame)
        if label:
            label = label.replace('_',' ').title()
        blink_detections = []
        is_detected = label is not None
    
    emit('blink_status', {'person': 'unknown' if label is None else label, 'is_detected':is_detected})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)