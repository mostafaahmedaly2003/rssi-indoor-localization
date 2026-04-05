"""
Flask API for RSSI-Based Indoor Localization
Usage: python app.py
Then POST to: http://localhost:5000/predict
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from src.config import MODEL_DIR, LOCATION_NAMES

app = Flask(__name__)

# Load model and scaler at startup
MODEL_NAME = "knn"  # change to mlp or svm if preferred
model  = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "RSSI Indoor Localization",
        "endpoints": {
            "GET  /": "This help message",
            "POST /predict": "Predict location from RSSI values",
            "GET  /locations": "List all reference locations",
        }
    })

@app.route("/locations", methods=["GET"])
def locations():
    return jsonify(LOCATION_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "rssi" not in data:
        return jsonify({"error": "Send JSON body: {\"rssi\": [-60, -55, -70, -65, -50, -75]}"}), 400
    rssi = data["rssi"]
    if len(rssi) != 6:
        return jsonify({"error": "Need exactly 6 RSSI values (one per AP)"}), 400
    X = np.array(rssi).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_id = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0].tolist()
    return jsonify({
        "predicted_location_id": pred_id,
        "predicted_location_name": LOCATION_NAMES[pred_id],
        "confidence": round(max(proba) * 100, 1),
        "all_probabilities": {
            LOCATION_NAMES[i]: round(p * 100, 1) for i, p in enumerate(proba)
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
