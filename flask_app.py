"""
flask_app.py
------------
Flask REST API and Web Application Server for Customer Churn Prediction and Risk Analysis.
Includes full OpenAPI / Swagger documentation and healthcheck endpoints.
"""

import os
import json
import time
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flasgger import Swagger, swag_from

from src.preprocessing import prepare_input_features
from src.predictor import (
    load_churn_model,
    predict_customer_churn,
    analyze_feature_contributions,
    generate_business_recommendations
)
from src.batch_processor import process_batch_csv

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configure Flasgger / OpenAPI Specification
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Telecom Customer Churn Prediction & Risk Analysis API",
        "description": "Production REST API for evaluating telecom customer churn risk, feature attribution drivers, retention recommendations, and batch CSV scoring.",
        "version": "1.0.0",
        "contact": {
            "name": "Telecom ML Team",
            "url": "https://github.com/SANJAY2334/Customer_churn_ML"
        }
    },
    "tags": [
        {"name": "Prediction", "description": "Single customer prediction and risk scoring"},
        {"name": "Batch Processing", "description": "Batch CSV file scoring and revenue-at-risk analysis"},
        {"name": "Model Intelligence", "description": "Model metrics, comparison curves, and evaluation stats"},
        {"name": "System", "description": "Application health and status"}
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

APP_START_TIME = time.time()
MODEL_PATH = "Models/churn_model.pkl"

# Load model lazily or at startup
try:
    model = load_churn_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Model could not be loaded on startup ({e}). Run train_models.py first.")

@app.route("/")
def home():
    """
    Web Dashboard UI
    ---
    tags:
      - System
    summary: Renders the web application dashboard interface.
    responses:
      200:
        description: HTML web page rendered successfully.
    """
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Service Health Check
    ---
    tags:
      - System
    summary: Returns system health, model availability, and runtime statistics.
    responses:
      200:
        description: System is healthy and operational.
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            uptime_seconds:
              type: number
              example: 45.2
            model_loaded:
              type: boolean
              example: true
            model_path:
              type: string
              example: Models/churn_model.pkl
      503:
        description: Model artifact is missing or not loaded.
    """
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = load_churn_model(MODEL_PATH)
        except Exception:
            pass

    is_model_ready = model is not None
    status_code = 200 if is_model_ready else 503

    return jsonify({
        "status": "healthy" if is_model_ready else "degraded",
        "uptime_seconds": round(time.time() - APP_START_TIME, 2),
        "model_loaded": is_model_ready,
        "model_path": MODEL_PATH
    }), status_code

@app.route("/predict", methods=["POST"])
def predict():
    """
    Single Customer Churn Prediction & Risk Analysis
    ---
    tags:
      - Prediction
    summary: Evaluates customer attributes and predicts churn probability, risk level, action, and drivers.
    consumes:
      - application/json
      - application/x-www-form-urlencoded
    parameters:
      - in: body
        name: body
        required: true
        description: Customer demographic, service, and billing profile.
        schema:
          type: object
          required:
            - tenure
            - monthly_charges
          properties:
            tenure:
              type: number
              example: 12
              description: Total account tenure in months
            monthly_charges:
              type: number
              example: 79.85
              description: Customer monthly recurring charges in USD
            total_charges:
              type: number
              example: 958.20
              description: Total charges billed to date (optional, imputed if omitted)
            gender:
              type: string
              enum: [Male, Female]
              example: Female
            senior_citizen:
              type: string
              enum: [Yes, No]
              example: No
            partner:
              type: string
              enum: [Yes, No]
              example: No
            dependents:
              type: string
              enum: [Yes, No]
              example: No
            phone_service:
              type: string
              enum: [Yes, No]
              example: Yes
            multiple_lines:
              type: string
              enum: [Yes, No, No phone service]
              example: No
            internet_service:
              type: string
              enum: [DSL, Fiber optic, No]
              example: Fiber optic
            online_security:
              type: string
              enum: [Yes, No, No internet service]
              example: No
            online_backup:
              type: string
              enum: [Yes, No, No internet service]
              example: No
            device_protection:
              type: string
              enum: [Yes, No, No internet service]
              example: No
            tech_support:
              type: string
              enum: [Yes, No, No internet service]
              example: No
            streaming_tv:
              type: string
              enum: [Yes, No, No internet service]
              example: No
            streaming_movies:
              type: string
              enum: [Yes, No, No internet service]
              example: No
            contract:
              type: string
              enum: [Month-to-month, One year, Two year]
              example: Month-to-month
            paperless_billing:
              type: string
              enum: [Yes, No]
              example: Yes
            payment_method:
              type: string
              enum: [Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)]
              example: Electronic check
    responses:
      200:
        description: Successful churn prediction and risk assessment.
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            prediction:
              type: string
              example: Likely to Churn
            churn_probability:
              type: number
              example: 74.5
            retain_probability:
              type: number
              example: 25.5
            risk_level:
              type: string
              example: High Risk
            risk_color:
              type: string
              example: "#ef4444"
            risk_action:
              type: string
              example: "🔴 Retention offer - Immediate Customer Success contact within 24 hours"
            recommendations:
              type: array
              items:
                type: string
            top_drivers:
              type: array
              items:
                type: object
                properties:
                  feature:
                    type: string
                    example: Contract_Month-to-month
                  impact:
                    type: number
                    example: 0.82
                  direction:
                    type: string
                    example: Increases Churn
      400:
        description: Invalid request payload or missing features.
      500:
        description: Model not initialized.
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_churn_model(MODEL_PATH)
        else:
            return jsonify({"status": "error", "message": "Model file not found. Please train model first."}), 500

    try:
        data = request.get_json(force=True, silent=True) if request.is_json else request.form.to_dict()
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400

        # Validate numeric inputs
        try:
            tenure_val = float(data.get('tenure', 0))
            monthly_charges_val = float(data.get('monthly_charges', 0.0))
        except (ValueError, TypeError) as e:
            return jsonify({"status": "error", "message": f"Invalid numerical value: {str(e)}"}), 400

        # Prepare input dict
        inputs = {
            'tenure': tenure_val,
            'monthly_charges': monthly_charges_val,
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

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    """
    Batch Customer Scoring from CSV
    ---
    tags:
      - Batch Processing
    summary: Uploads a CSV file of customer records and scores churn probability, risk level, CLV, and revenue at risk.
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: CSV file containing customer records matching IBM Telco features.
    responses:
      200:
        description: Batch scoring completed successfully.
        schema:
          type: object
          properties:
            status:
              type: string
              example: success
            summary:
              type: object
              properties:
                total_records:
                  type: integer
                  example: 500
                churn_rate:
                  type: number
                  example: 26.8
                high_risk_count:
                  type: integer
                  example: 134
                total_revenue_at_risk_$:
                  type: number
                  example: 48920.50
            preview:
              type: array
              items:
                type: object
      400:
        description: Missing file or invalid CSV contents.
      500:
        description: Model not loaded.
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_churn_model(MODEL_PATH)
        else:
            return jsonify({"status": "error", "message": "Model file not found."}), 500

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded in form field 'file'"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400

    try:
        df_input = pd.read_csv(file)
        if df_input.empty:
            return jsonify({"status": "error", "message": "Uploaded CSV file is empty."}), 400

        df_scored, summary = process_batch_csv(df_input, model)
        
        # Convert first 20 records for preview
        records = df_scored.head(20).to_dict(orient="records")

        return jsonify({
            "status": "success",
            "summary": summary,
            "preview": records
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Model Comparison Metrics & ROC/PR Curves
    ---
    tags:
      - Model Intelligence
    summary: Returns evaluation metrics, confusion matrices, and ROC/PR curve points for all trained classifiers.
    responses:
      200:
        description: Model evaluation metrics JSON.
      404:
        description: Metrics file not found.
    """
    metrics_path = "Models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data), 200
    else:
        return jsonify({"status": "error", "message": "Metrics file not found. Run train_models.py first."}), 404

if __name__ == "__main__":
    print("Starting Flask Server for Customer Churn Prediction on http://127.0.0.1:5000")
    print("Interactive Swagger API Documentation available at http://127.0.0.1:5000/apidocs/")
    app.run(host="127.0.0.1", port=5000, debug=False)
