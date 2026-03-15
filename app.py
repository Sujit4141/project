from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os
import math
import base64

app = Flask(__name__)
CORS(app)  # must be right after app = Flask(__name__)

# ----------------------------
# Load ML model
# ----------------------------
try:
    model = joblib.load("fatigue_model.pkl")
except FileNotFoundError:
    raise Exception("fatigue_model.pkl not found.")

# ----------------------------
# Download MediaPipe Face Landmarker model
# ----------------------------
LANDMARK_MODEL_PATH = "face_landmarker.task"
if not os.path.exists(LANDMARK_MODEL_PATH):
    print("Downloading Face Landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        LANDMARK_MODEL_PATH
    )
    print("Downloaded.")

# ----------------------------
# Setup options for Face Landmarker (do not create instance here)
# ----------------------------
base_options = mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
landmarker_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize face_landmarker variable as None; will be created on first use
face_landmarker = None

# ----------------------------
# Eye landmark indices
# ----------------------------
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD    = 0.21
BLINK_CONSEC_FRAMES = 2

def compute_ear(landmarks, eye_indices, img_w, img_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def compute_head_pose(transformation_matrix):
    mat = np.array(transformation_matrix.data).reshape(4, 4)
    R = mat[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        roll  = 0
    return pitch, yaw, roll

def analyze_frame(frame):
    global face_landmarker  # use global so we can assign below

    # Create face_landmarker on first use to avoid startup crash
    if face_landmarker is None:
        face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return False, 0, 0, 0, 0, 0, 0, 0

    landmarks = results.face_landmarks[0]
    left_ear  = compute_ear(landmarks, LEFT_EYE,  w, h)
    right_ear = compute_ear(landmarks, RIGHT_EYE, w, h)

    blink_left  = 0
    blink_right = 0
    if results.face_blendshapes:
        for bs in results.face_blendshapes[0]:
            if bs.category_name == "eyeBlinkLeft":
                blink_left = bs.score
            elif bs.category_name == "eyeBlinkRight":
                blink_right = bs.score

    pitch, yaw, roll = 0, 0, 0
    if results.facial_transformation_matrixes:
        pitch, yaw, roll = compute_head_pose(results.facial_transformation_matrixes[0])

    return True, left_ear, right_ear, pitch, yaw, roll, blink_left, blink_right


# ----------------------------
# NEW: Analyze a single base64 frame from React browser webcam
# ----------------------------
@app.route("/analyze-frame", methods=["POST", "OPTIONS"])
def analyze_frame_route():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.json
    img_b64 = data.get("image", "")

    try:
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    (face_detected, left_ear, right_ear,
     pitch, yaw, roll, blink_left, blink_right) = analyze_frame(frame)

    avg_ear    = (left_ear + right_ear) / 2.0
    eyes_closed = avg_ear < EAR_THRESHOLD and face_detected

    return jsonify({
        "face_detected":     face_detected,
        "left_ear":          round(left_ear,  3),
        "right_ear":         round(right_ear, 3),
        "avg_ear":           round(avg_ear,   3),
        "eyes_closed":       eyes_closed,
        "pitch":             round(pitch, 2),
        "yaw":               round(yaw,   2),
        "roll":              round(roll,  2),
        "blink_score_left":  round(blink_left,  3),
        "blink_score_right": round(blink_right, 3),
    })


# ----------------------------
# NEW: Final prediction after React session ends
# ----------------------------
@app.route("/predict-session", methods=["POST", "OPTIONS"])
def predict_session():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.json

    sample = pd.DataFrame([{
        "blink_rate":       data.get("blink_rate",       0),
        "eye_closure_time": data.get("eye_closure_time", 0),
        "head_tilt_angle":  data.get("head_tilt_angle",  0),
        "heart_rate":       data.get("heart_rate",       95),
        "shift_hours":      data.get("shift_hours",      5),
        "temperature":      data.get("temperature",      32),
        "gas_level":        data.get("gas_level",        0.03),
    }])

    prediction = model.predict(sample)
    levels = ["Normal", "Moderate", "High"]
    fatigue_level = levels[int(prediction[0])]

    return jsonify({
        "fatigue_level":    fatigue_level,
        "blink_rate":       round(data.get("blink_rate", 0), 2),
        "eye_closure_time": round(data.get("eye_closure_time", 0), 2),
        "head_tilt_angle":  round(data.get("head_tilt_angle", 0), 2),
    })


# ----------------------------
# Original: Manual prediction route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sample = pd.DataFrame([{
        "blink_rate":       data.get("blink_rate", 0),
        "eye_closure_time": data.get("eye_closure_time", 0),
        "head_tilt_angle":  data.get("head_tilt_angle", 0),
        "heart_rate":       data.get("heart_rate", 0),
        "shift_hours":      data.get("shift_hours", 0),
        "temperature":      data.get("temperature", 0),
        "gas_level":        data.get("gas_level", 0)
    }])
    prediction = model.predict(sample)
    levels = ["Normal", "Moderate", "High"]
    return jsonify({"fatigue_level": levels[int(prediction[0])]})

@app.route("/")
def home():
    return "Fatigue Detection API Running"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
