"""
Synthetic Crop Dataset Generator + Model Trainer
=================================================
Based on published agronomic literature for tropical/subtropical conditions.

Sources:
- FAO Plant Nutrition for Food Security (2006)
- ICAR Crop Production Guide (India)
- Tamil Nadu Agricultural University (TNAU) Crop Management Guidelines
- Doorenbos & Kassam, FAO Irrigation and Drainage Paper No. 33
Crops: Spinach, Tomato, Chilli (Hot Pepper), Lady Finger (Okra)
Features: N, P, K (mg/kg), Temperature (°C), Humidity (%), pH, Rainfall (mm)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import json

np.random.seed(42)

# ─── Agronomic Parameter Ranges ─────────────────────────────────
# Format: (mean, std) — values drawn from normal distribution
# then clipped to realistic min/max

CROP_PROFILES = {
    "spinach": {
        # Spinach: cool season leafy, moderate N, low P/K
        # Ref: TNAU Agritech Portal, Spinach cultivation guide
        "N":           (120, 20),    # mg/kg, needs moderate nitrogen for leaf growth
        "P":           (55,  10),    # mg/kg, moderate phosphorus
        "K":           (200, 25),    # mg/kg, high potassium for quality
        "temperature": (22,   3),    # °C, prefers cooler temps 18-28°C
        "humidity":    (65,   8),    # %, moderate humidity
        "ph":          (6.5,  0.3),  # slightly acidic to neutral
        "rainfall":    (80,  20),    # mm, moderate water requirement
        "clip": {
            "N": (80, 160), "P": (30, 80), "K": (150, 260),
            "temperature": (15, 30), "humidity": (50, 80),
            "ph": (5.5, 7.5), "rainfall": (40, 140)
        }
    },
    "tomato": {
        # Tomato: warm season fruiting, high N/P/K demand
        # Ref: FAO Crop Water Requirements, ICAR Tomato Production
        "N":           (180, 25),    # mg/kg, high nitrogen for fruit development
        "P":           (80,  15),    # mg/kg, high phosphorus for root/fruit
        "K":           (250, 30),    # mg/kg, very high potassium for fruit quality
        "temperature": (28,   3),    # °C, warm crop 22-32°C
        "humidity":    (60,   8),    # %, moderate, too high causes blight
        "ph":          (6.2,  0.4),  # slightly acidic
        "rainfall":    (120, 25),    # mm, high water requirement
        "clip": {
            "N": (130, 240), "P": (50, 120), "K": (190, 320),
            "temperature": (20, 35), "humidity": (45, 75),
            "ph": (5.5, 7.0), "rainfall": (60, 180)
        }
    },
    "chilli": {
        # Chilli (Hot Pepper): warm, drought tolerant, moderate nutrients
        # Ref: TNAU Chilli Cultivation Guide, FAO Pepper Production
        "N":           (150, 20),    # mg/kg, moderate nitrogen
        "P":           (70,  12),    # mg/kg, moderate phosphorus
        "K":           (220, 25),    # mg/kg, high potassium for pungency
        "temperature": (30,   3),    # °C, hot crop 25-35°C
        "humidity":    (55,   8),    # %, prefers drier conditions
        "ph":          (6.0,  0.4),  # slightly acidic
        "rainfall":    (90,  20),    # mm, moderate, drought tolerant
        "clip": {
            "N": (100, 200), "P": (40, 100), "K": (170, 280),
            "temperature": (22, 38), "humidity": (40, 72),
            "ph": (5.0, 7.0), "rainfall": (50, 140)
        }
    },
    "ladyfinger": {
        # Lady Finger (Okra): tropical, heat loving, moderate-high nutrients
        # Ref: ICAR Okra Production Technology, TNAU Bhendi Guide
        "N":           (160, 22),    # mg/kg, moderate-high nitrogen
        "P":           (65,  12),    # mg/kg, moderate phosphorus
        "K":           (235, 28),    # mg/kg, high potassium
        "temperature": (32,   3),    # °C, heat loving 28-38°C
        "humidity":    (70,   8),    # %, tolerates high humidity
        "ph":          (6.4,  0.4),  # near neutral
        "rainfall":    (110, 22),    # mm, moderate-high
        "clip": {
            "N": (110, 210), "P": (35, 95), "K": (175, 300),
            "temperature": (24, 40), "humidity": (55, 85),
            "ph": (5.5, 7.5), "rainfall": (60, 160)
        }
    }
}

# ─── Generate Synthetic Samples ──────────────────────────────────
SAMPLES_PER_CROP = 500  # 2000 total samples
rows = []

for crop, profile in CROP_PROFILES.items():
    clip = profile["clip"]
    for _ in range(SAMPLES_PER_CROP):
        row = {}
        for feat in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            mean, std = profile[feat]
            val = np.random.normal(mean, std)
            val = np.clip(val, clip[feat][0], clip[feat][1])
            if feat in ["N", "P", "K"]:
                val = round(val)
            else:
                val = round(val, 2)
            row[feat] = val
        row["label"] = crop
        rows.append(row)

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Generated dataset: {len(df)} samples")
print(f"Crops: {df['label'].value_counts().to_dict()}")
print(f"\nDataset summary (means):")
print(df.groupby("label")[["N","P","K","temperature","humidity"]].mean().round(1))

df.to_csv("synthetic_crop_data.csv", index=False)
print("\nSaved: synthetic_crop_data.csv")

# ─── Train Model ─────────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES].values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")

print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# ─── Evaluate ────────────────────────────────────────────────────
y_pred    = model.predict(X_test)
acc       = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y_encoded, cv=5)

print(f"\n{'='*55}")
print(f"  Test Accuracy:             {acc*100:.2f}%")
print(f"  Cross-validation (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"{'='*55}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

print("\nFeature Importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:15s}: {imp*100:.1f}%")

# ─── Save Model & Metadata ───────────────────────────────────────
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

crop_stats = {}
for crop in le.classes_:
    subset = df[df["label"] == crop]
    display = {"spinach": "Spinach", "tomato": "Tomato", "chilli": "Chilli", "ladyfinger": "Lady Finger"}
    crop_stats[crop] = {
        "display_name": display.get(crop, crop.title()),
        "N":           {"min": int(subset["N"].min()),                "max": int(subset["N"].max()),                "mean": round(float(subset["N"].mean()), 1)},
        "P":           {"min": int(subset["P"].min()),                "max": int(subset["P"].max()),                "mean": round(float(subset["P"].mean()), 1)},
        "K":           {"min": int(subset["K"].min()),                "max": int(subset["K"].max()),                "mean": round(float(subset["K"].mean()), 1)},
        "temperature": {"min": round(float(subset["temperature"].min()), 1), "max": round(float(subset["temperature"].max()), 1), "mean": round(float(subset["temperature"].mean()), 1)},
        "humidity":    {"min": round(float(subset["humidity"].min()), 1),    "max": round(float(subset["humidity"].max()), 1),    "mean": round(float(subset["humidity"].mean()), 1)},
        "ph":          {"min": round(float(subset["ph"].min()), 1),          "max": round(float(subset["ph"].max()), 1),          "mean": round(float(subset["ph"].mean()), 1)},
        "advice": {
            "N_low":     f"Apply nitrogen-rich fertilizer (urea or ammonium sulphate) to boost {crop} growth.",
            "N_high":    f"Reduce nitrogen input — excess causes leafy growth at the expense of yield in {crop}.",
            "P_low":     f"Apply superphosphate or bone meal to improve root development in {crop}.",
            "K_low":     f"Apply potassium sulphate to improve {crop} fruit quality and disease resistance.",
            "temp_high": f"Provide shade netting — {crop} is experiencing heat stress above its optimal range.",
            "temp_low":  f"Protect {crop} from cold — consider row covers or relocating the grow bag indoors.",
        }
    }

with open("crop_stats.json", "w") as f:
    json.dump(crop_stats, f, indent=2)

with open("model_features.json", "w") as f:
    json.dump(FEATURES, f)

print("\nSaved: crop_model.pkl, label_encoder.pkl, crop_stats.json, model_features.json")
print("\nAll done! Model is ready for Flask backend.")
