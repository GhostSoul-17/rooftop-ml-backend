"""
Rooftop Horticulture DSS — Flask Backend
=========================================
Fetches live sensor data from Firebase, runs ML crop recommendation,
and pushes prediction results back to Firebase for the dashboard.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import json
import requests
import numpy as np
from threading import Thread
import time

app = Flask(__name__)
CORS(app)

# ─── Firebase Config ────────────────────────────────────────────
FIREBASE_BASE = "https://rooftop-horticulture-default-rtdb.asia-southeast1.firebasedatabase.app"
DB_SECRET     = "fYdJU6CJofeGhkz9zZA5mkIbN1M9YNzLUnkmLiie"  # replace with your actual secret

# ─── Load Model & Metadata ──────────────────────────────────────
print("Loading ML model...")
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("crop_stats.json", "r") as f:
    crop_stats = json.load(f)

with open("model_features.json", "r") as f:
    FEATURES = json.load(f)

print(f"Model loaded. Classes: {list(le.classes_)}")
print(f"Features: {FEATURES}")

# ─── Helper: Get sensor data from Firebase ──────────────────────
def get_sensor_data():
    try:
        url = f"{FIREBASE_BASE}/sensor/latest.json?auth={DB_SECRET}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Firebase read error: {e}")
    return None

# ─── Helper: Push prediction to Firebase ────────────────────────
def push_prediction(prediction_data):
    try:
        url = f"{FIREBASE_BASE}/ml/prediction.json?auth={DB_SECRET}"
        res = requests.put(url, json=prediction_data, timeout=10)
        if res.status_code == 200:
            print("Prediction pushed to Firebase.")
        else:
            print(f"Firebase write error: {res.status_code}")
    except Exception as e:
        print(f"Firebase push error: {e}")

# ─── Core Prediction Function ────────────────────────────────────
def run_prediction(sensor_data):
    """
    Maps sensor readings to model features and runs prediction.
    
    Sensor   → Model Feature mapping:
    N        → N
    P        → P  
    K        → K
    temp     → temperature
    moisture → humidity  (soil moisture approximates ambient humidity)
    EC       → ph        (EC inversely approximates pH — high EC = lower pH)
    light    → rainfall  (light intensity approximates available solar energy)
    """

    # Extract sensor values with safe defaults
    N           = float(sensor_data.get("nitrogen",    0))
    P           = float(sensor_data.get("phosphorus",  0))
    K           = float(sensor_data.get("potassium",   0))
    temperature = float(sensor_data.get("temperature", 25))
    moisture    = float(sensor_data.get("moisture",    50))
    ec          = float(sensor_data.get("ec",          500))
    light       = float(sensor_data.get("light",       500))

    # Map EC to approximate pH (EC 0-500 → pH ~7.5, EC 500-2000 → pH ~6.5-5.5)
    if ec <= 0:
        ph = 6.5  # neutral default
    else:
        ph = max(5.0, min(7.5, 7.5 - (ec / 1000)))

    # Map light (lux) to approximate rainfall equivalent (mm)
    # 0 lux = 0mm, 10000+ lux = 150mm equivalent
    rainfall = min(150, light / 70)

    # Build feature vector in correct order
    features = np.array([[N, P, K, temperature, moisture, ph, rainfall]])

    # Get prediction probabilities for all crops
    proba = model.predict_proba(features)[0]
    predicted_idx = np.argmax(proba)
    predicted_crop = le.classes_[predicted_idx]

    # Build suitability scores for all crops
    suitability = {}
    for i, crop in enumerate(le.classes_):
        score = round(float(proba[i]) * 100, 1)
        suitability[crop] = score

    # Get crop-specific advice based on current readings
    advice = []
    stats  = crop_stats.get(predicted_crop, {})

    if N > 0:
        if N < stats.get("N", {}).get("min", 0):
            advice.append(stats.get("advice", {}).get("N_low", ""))
        elif N > stats.get("N", {}).get("max", 999):
            advice.append(stats.get("advice", {}).get("N_high", ""))

    if P > 0 and P < stats.get("P", {}).get("min", 0):
        advice.append(stats.get("advice", {}).get("P_low", ""))

    if K > 0 and K < stats.get("K", {}).get("min", 0):
        advice.append(stats.get("advice", {}).get("K_low", ""))

    if temperature > stats.get("temperature", {}).get("max", 40):
        advice.append(stats.get("advice", {}).get("temp_high", ""))
    elif temperature < stats.get("temperature", {}).get("min", 0):
        advice.append(stats.get("advice", {}).get("temp_low", ""))

    if not advice:
        advice.append(f"Current conditions are suitable for {predicted_crop}. Maintain current practices.")

    # Confidence label
    confidence = proba[predicted_idx]
    if confidence >= 0.75:
        confidence_label = "High"
    elif confidence >= 0.50:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low"

    return {
        "predicted_crop":  predicted_crop,
        "confidence":      round(float(confidence) * 100, 1),
        "confidence_label": confidence_label,
        "suitability":     suitability,
        "advice":          [a for a in advice if a],
        "input_features": {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "moisture": moisture,
            "ec": ec,
            "light": light,
            "ph_estimated": round(ph, 2),
            "rainfall_estimated": round(rainfall, 2)
        }
    }

# ─── Background Loop ─────────────────────────────────────────────
def prediction_loop():
    print("Starting prediction loop — runs every 30 seconds...")
    while True:
        try:
            sensor_data = get_sensor_data()
            if sensor_data:
                result = run_prediction(sensor_data)
                result["timestamp"] = sensor_data.get("timestamp", "")
                push_prediction(result)
                print(f"Predicted: {result['predicted_crop']} ({result['confidence']}% confidence)")
            else:
                print("No sensor data available.")
        except Exception as e:
            print(f"Prediction loop error: {e}")
        time.sleep(30)

# ─── API Routes ──────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Rooftop Horticulture DSS — ML Backend",
        "crops": list(le.classes_),
        "features": FEATURES
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Manual prediction endpoint — accepts JSON sensor data"""
    try:
        data   = request.get_json()
        result = run_prediction(data)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/live", methods=["GET"])
def predict_live():
    """Fetch latest sensor data from Firebase and predict"""
    try:
        sensor_data = get_sensor_data()
        if not sensor_data:
            return jsonify({"status": "error", "message": "No sensor data in Firebase"}), 404
        result = run_prediction(sensor_data)
        result["timestamp"] = sensor_data.get("timestamp", "")
        push_prediction(result)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/crops", methods=["GET"])
def get_crops():
    """Return crop profiles and stats"""
    return jsonify({"status": "success", "crops": crop_stats})

# ─── Start App ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Start background prediction loop in a separate thread
    thread = Thread(target=prediction_loop, daemon=True)
    thread.start()

    print("Flask server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
