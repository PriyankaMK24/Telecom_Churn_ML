"""
test_api.py
-----------
Integration tests for Flask REST API endpoints and Swagger specifications.
"""

import io
import json
import pytest

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Customer Churn" in response.data

def test_swagger_ui_and_spec(client):
    ui_resp = client.get("/apidocs/")
    assert ui_resp.status_code == 200

    spec_resp = client.get("/apispec_1.json")
    assert spec_resp.status_code == 200
    spec_data = json.loads(spec_resp.data)
    assert "swagger" in spec_data or "openapi" in spec_data
    assert "/predict" in spec_data["paths"]
    assert "/api/health" in spec_data["paths"]

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data

def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "best_model" in data
    assert "metrics" in data

def test_predict_success(client, sample_customer_high_risk):
    response = client.post(
        "/predict",
        data=json.dumps(sample_customer_high_risk),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert "prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "risk_action" in data
    assert "recommendations" in data
    assert "top_drivers" in data

def test_predict_invalid_payload(client):
    response = client.post(
        "/predict",
        data=json.dumps({"tenure": "not_a_number", "monthly_charges": "abc"}),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["status"] == "error"

def test_batch_predict_success(client):
    # Create sample CSV in memory
    csv_content = (
        "Tenure Months,Monthly Charges,Total Charges,Gender,Senior Citizen,Partner,Dependents,"
        "Phone Service,Multiple Lines,Internet Service,Online Security,Online Backup,Device Protection,"
        "Tech Support,Streaming TV,Streaming Movies,Contract,Paperless Billing,Payment Method\n"
        "1,70.5,70.5,Female,No,No,No,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check\n"
        "48,20.0,960.0,Male,No,Yes,Yes,Yes,No,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Two year,No,Mailed check\n"
    )
    data = {
        'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_customers.csv')
    }
    response = client.post(
        "/api/batch_predict",
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    res_data = json.loads(response.data)
    assert res_data["status"] == "success"
    assert res_data["summary"]["total_records"] == 2
    assert len(res_data["preview"]) == 2

def test_batch_predict_missing_file(client):
    response = client.post("/api/batch_predict", data={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["status"] == "error"
