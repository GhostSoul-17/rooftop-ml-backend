"""
Rooftop Horticulture DSS — Flask Backend
=========================================
- Fetches live sensor data from Firebase
- Runs ML crop recommendation
- Fetches weather data from OpenWeatherMap
- Pushes all results back to Firebase
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import json
import requests
import numpy as np
from threading import Thread
import time
import os

app = Flask(__name__)
CORS(app)

# ─── Config ─────────────────────────────────────────────────────
FIREBASE_BASE   = "https://rooftop-horticulture-default-rtdb.asia-southeast1.firebasedatabase.app"
DB_SECRET       = os.environ.get("DB_SECRET", "your-secret-key-here")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "b42c4437194a42406695832949c70ad5")
LAT             = 13.0827   # Chennai
LON             = 80.2707   # Chennai

# ─── Load Model ─────────────────────────────────────────────────
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

# ─── Firebase Helpers ────────────────────────────────────────────
def firebase_get(path):
    try:
        url = f"{FIREBASE_BASE}/{path}.json?auth={DB_SECRET}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Firebase GET error: {e}")
    return None

def firebase_put(path, data):
    try:
        url = f"{FIREBASE_BASE}/{path}.json?auth={DB_SECRET}"
        res = requests.put(url, json=data, timeout=10)
        return res.status_code == 200
    except Exception as e:
        print(f"Firebase PUT error: {e}")
    return False

# ─── Weather Functions ───────────────────────────────────────────
def fetch_weather():
    """Fetch current weather and 5-day forecast from OpenWeatherMap"""
    try:
        # Current weather
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}&units=metric"
        current_res = requests.get(current_url, timeout=10)

        # 5-day forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}&units=metric&cnt=8"
        forecast_res = requests.get(forecast_url, timeout=10)

        if current_res.status_code != 200 or forecast_res.status_code != 200:
            print(f"Weather API error: {current_res.status_code}, {forecast_res.status_code}")
            return None

        current  = current_res.json()
        forecast = forecast_res.json()

        # Extract current conditions
        current_data = {
            "temperature":  round(current["main"]["temp"], 1),
            "feels_like":   round(current["main"]["feels_like"], 1),
            "humidity":     current["main"]["humidity"],
            "pressure":     current["main"]["pressure"],
            "description":  current["weather"][0]["description"].title(),
            "icon":         current["weather"][0]["icon"],
            "wind_speed":   round(current["wind"]["speed"] * 3.6, 1),  # m/s to km/h
            "rain_1h":      current.get("rain", {}).get("1h", 0),
            "clouds":       current["clouds"]["all"],
            "visibility":   current.get("visibility", 10000),
        }

        # Extract next 24 hours forecast (8 x 3-hour intervals)
        forecast_items = []
        rain_expected  = False
        rain_amount    = 0
        max_temp_24h   = current_data["temperature"]

        for item in forecast["list"]:
            rain = item.get("rain", {}).get("3h", 0)
            rain_amount += rain
            if rain > 0:
                rain_expected = True
            if item["main"]["temp"] > max_temp_24h:
                max_temp_24h = item["main"]["temp"]

            forecast_items.append({
                "time":        item["dt_txt"].split(" ")[1][:5],
                "temp":        round(item["main"]["temp"], 1),
                "humidity":    item["main"]["humidity"],
                "description": item["weather"][0]["description"].title(),
                "icon":        item["weather"][0]["icon"],
                "rain":        round(rain, 1),
                "wind_speed":  round(item["wind"]["speed"] * 3.6, 1),
            })

        # DSS weather decision
        skip_irrigation = False
        weather_alert   = None
        weather_advice  = []

        if rain_expected and rain_amount > 5:
            skip_irrigation = True
            weather_advice.append(f"Rain expected in next 24 hours ({round(rain_amount, 1)}mm total) — irrigation can be skipped.")
            weather_alert = "rain"
        elif rain_expected:
            weather_advice.append(f"Light rain possible ({round(rain_amount, 1)}mm) — monitor soil moisture before irrigating.")
            weather_alert = "light_rain"

        if max_temp_24h > 35:
            weather_advice.append(f"High temperature alert — max {round(max_temp_24h, 1)}°C expected. Increase irrigation frequency and consider shade netting.")
            weather_alert = weather_alert or "heat"

        if current_data["wind_speed"] > 30:
            weather_advice.append(f"Strong winds ({current_data['wind_speed']} km/h) — secure grow bags and check for physical damage to plants.")

        if current_data["humidity"] > 85:
            weather_advice.append("High humidity — watch for fungal disease. Ensure good air circulation around plants.")

        if not weather_advice:
            weather_advice.append("Weather conditions are favourable for all 4 crops today.")

        return {
            "current":          current_data,
            "forecast":         forecast_items,
            "rain_expected":    rain_expected,
            "rain_amount_24h":  round(rain_amount, 1),
            "max_temp_24h":     round(max_temp_24h, 1),
            "skip_irrigation":  skip_irrigation,
            "weather_alert":    weather_alert,
            "weather_advice":   weather_advice,
            "location":         "Chennai, IN",
            "last_updated":     time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"Weather fetch error: {e}")
        return None

# ─── ML Prediction ───────────────────────────────────────────────
def run_prediction(sensor_data):
    N           = float(sensor_data.get("nitrogen",    0))
    P           = float(sensor_data.get("phosphorus",  0))
    K           = float(sensor_data.get("potassium",   0))
    temperature = float(sensor_data.get("temperature", 25))
    moisture    = float(sensor_data.get("moisture",    50))
    ec          = float(sensor_data.get("ec",          500))
    light       = float(sensor_data.get("light",       500))

    ph       = max(5.0, min(7.5, 7.5 - (ec / 1000))) if ec > 0 else 6.5
    rainfall = min(150, light / 70)

    features = np.array([[N, P, K, temperature, moisture, ph, rainfall]])
    proba    = model.predict_proba(features)[0]
    pred_idx = np.argmax(proba)
    pred_crop = le.classes_[pred_idx]

    suitability = {}
    for i, crop in enumerate(le.classes_):
        suitability[crop] = round(float(proba[i]) * 100, 1)

    advice = []
    stats  = crop_stats.get(pred_crop, {})
    if N > 0:
        if N < stats.get("N", {}).get("min", 0):   advice.append(stats.get("advice", {}).get("N_low", ""))
        elif N > stats.get("N", {}).get("max", 999): advice.append(stats.get("advice", {}).get("N_high", ""))
    if P > 0 and P < stats.get("P", {}).get("min", 0): advice.append(stats.get("advice", {}).get("P_low", ""))
    if K > 0 and K < stats.get("K", {}).get("min", 0): advice.append(stats.get("advice", {}).get("K_low", ""))
    if temperature > stats.get("temperature", {}).get("max", 40): advice.append(stats.get("advice", {}).get("temp_high", ""))
    elif temperature < stats.get("temperature", {}).get("min", 0): advice.append(stats.get("advice", {}).get("temp_low", ""))
    if not advice:
        advice.append(f"Current conditions are suitable for {pred_crop}. Maintain current practices.")

    confidence = proba[pred_idx]
    conf_label = "High" if confidence >= 0.75 else "Moderate" if confidence >= 0.50 else "Low"

    return {
        "predicted_crop":   pred_crop,
        "confidence":       round(float(confidence) * 100, 1),
        "confidence_label": conf_label,
        "suitability":      suitability,
        "advice":           [a for a in advice if a],
        "input_features": {
            "N": N, "P": P, "K": K,
            "temperature": temperature, "moisture": moisture,
            "ec": ec, "light": light,
            "ph_estimated": round(ph, 2),
            "rainfall_estimated": round(rainfall, 2)
        }
    }

# ─── Background Loop ─────────────────────────────────────────────
def background_loop():
    print("Background loop started...")
    weather_counter = 0

    while True:
        try:
            # ML prediction every 30 seconds
            sensor_data = firebase_get("sensor/latest")
            if sensor_data:
                result = run_prediction(sensor_data)
                result["timestamp"] = sensor_data.get("timestamp", "")
                firebase_put("ml/prediction", result)
                print(f"ML: {result['predicted_crop']} ({result['confidence']}%)")

            # Weather every 10 minutes (every 20 loops of 30 seconds)
            weather_counter += 1
            if weather_counter >= 20:
                weather_counter = 0
                weather_data = fetch_weather()
                if weather_data:
                    firebase_put("weather", weather_data)
                    print(f"Weather: {weather_data['current']['description']}, rain={weather_data['rain_expected']}")

        except Exception as e:
            print(f"Background loop error: {e}")

        time.sleep(30)

# ─── API Routes ──────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":   "running",
        "message":  "Rooftop Horticulture DSS — ML + Weather Backend",
        "crops":    list(le.classes_),
        "features": FEATURES
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json()
        result = run_prediction(data)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/live", methods=["GET"])
def predict_live():
    try:
        sensor_data = firebase_get("sensor/latest")
        if not sensor_data:
            return jsonify({"status": "error", "message": "No sensor data"}), 404
        result = run_prediction(sensor_data)
        result["timestamp"] = sensor_data.get("timestamp", "")
        firebase_put("ml/prediction", result)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/weather", methods=["GET"])
def weather():
    try:
        data = fetch_weather()
        if data:
            firebase_put("weather", data)
            return jsonify({"status": "success", "weather": data})
        return jsonify({"status": "error", "message": "Weather fetch failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/crops", methods=["GET"])
def get_crops():
    return jsonify({"status": "success", "crops": crop_stats})

# ─── Start ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Fetch weather immediately on startup
    print("Fetching initial weather data...")
    weather_data = fetch_weather()
    if weather_data:
        firebase_put("weather", weather_data)
        print(f"Weather ready: {weather_data['current']['description']}")
    else:
        print("Weather fetch failed — check API key")

    thread = Thread(target=background_loop, daemon=True)
    thread.start()

    print("Flask server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)