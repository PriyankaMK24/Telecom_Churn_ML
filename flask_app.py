"""
flask_app.py
------------
Flask REST API and Web Application Server for Customer Churn Prediction and Risk Analysis.
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.preprocessing import prepare_input_features
from src.predictor import (
    load_churn_model,
    predict_customer_churn,
    analyze_feature_contributions,
    generate_business_recommendations
)
from src.batch_processor import process_batch_csv

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model lazily or at startup
MODEL_PATH = "Models/churn_model.pkl"
try:
    model = load_churn_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Model could not be loaded on startup ({e}). Run train_models.py first.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_churn_model(MODEL_PATH)
        else:
            return jsonify({"error": "Model file not found. Please train model first."}), 500

    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        
        # Prepare input dict
        inputs = {
            'tenure': float(data.get('tenure', 0)),
            'monthly_charges': float(data.get('monthly_charges', 0.0)),
            'total_charges': data.get('total_charges', None),
            'gender': data.get('gender', 'Female'),
            'senior_citizen': data.get('senior_citizen', 'No'),
            'partner': data.get('partner', 'No'),
            'dependents': data.get('dependents', 'No'),
            'phone_service': data.get('phone_service', 'Yes'),
            'multiple_lines': data.get('multiple_lines', 'No'),
            'internet_service': data.get('internet_service', 'Fiber optic'),
            'online_security': data.get('online_security', 'No'),
            'online_backup': data.get('online_backup', 'No'),
            'device_protection': data.get('device_protection', 'No'),
            'tech_support': data.get('tech_support', 'No'),
            'streaming_tv': data.get('streaming_tv', 'No'),
            'streaming_movies': data.get('streaming_movies', 'No'),
            'contract': data.get('contract', 'Month-to-month'),
            'paperless_billing': data.get('paperless_billing', 'Yes'),
            'payment_method': data.get('payment_method', 'Electronic check')
        }

        # Encode input features
        df_encoded = prepare_input_features(inputs)

        # Predict churn
        pred_label, churn_proba, retain_proba, risk_level, risk_color = predict_customer_churn(model, df_encoded)

        # Blueprint risk action mapping
        if risk_level == "Low Risk":
            risk_action = "🟢 Normal service - Maintain current relationship & explore loyalty program"
        elif risk_level == "Medium Risk":
            risk_action = "🟡 Monitor customer - Send proactive engagement & renewal discount"
        else:
            risk_action = "🔴 Retention offer - Immediate Customer Success contact within 24 hours"

        # Generate business recommendations
        recs = generate_business_recommendations(inputs, churn_proba, risk_level)

        # Driver analysis
        top_drivers_df = analyze_feature_contributions(model, df_encoded, top_n=5)
        drivers_list = top_drivers_df.to_dict(orient="records")

        response = {
            "status": "success",
            "prediction": pred_label,
            "churn_probability": round(churn_proba * 100, 1),
            "retain_probability": round(retain_proba * 100, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_action": risk_action,
            "recommendations": recs,
            "top_drivers": drivers_list
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    global model
    if model is None:
        return jsonify({"error": "Model file not found."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        df_input = pd.read_csv(file)
        df_scored, summary = process_batch_csv(df_input, model)
        
        # Convert first 20 records for preview
        records = df_scored.head(20).to_dict(orient="records")

        return jsonify({
            "status": "success",
            "summary": summary,
            "preview": records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = "Models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics file not found. Run train_models.py first."}), 404

if __name__ == "__main__":
    print("Starting Flask Server for Customer Churn Prediction on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
