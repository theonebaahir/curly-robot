from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import sys

# Auto-install dependencies if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["flask", "opencv-python", "mediapipe", "numpy"]:
    try:
        __import__(pkg if pkg != "opencv-python" else "cv2")
    except ImportError:
        install(pkg)

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    reps = 0
    stage = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            def calculate_angle(a, b, c):
                a, b, c = np.array(a), np.array(b), np.array(c)
                radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)
                if angle > 180.0:
                    angle = 360-angle
                return angle

            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            if angle > 160:
                stage = "up"
            if angle < 60 and stage == "up":
                stage = "down"
                reps += 1

    cap.release()
    return reps

@app.route("/")
def home():
    return "FitBank Workout Verifier API is running!"

@app.route("/verify", methods=["POST"])
def verify():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    video_path = os.path.join("temp_video.mp4")
    video.save(video_path)

    reps = analyze_video(video_path)

    os.remove(video_path)

    return jsonify({"reps_detected": reps, "status": "Workout verified!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

