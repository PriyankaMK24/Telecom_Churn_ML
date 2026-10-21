"""
test_predictor.py
------------------
Unit tests for model inference, risk classification, feature drivers, and recommendations.
"""

import pandas as pd
from src.preprocessing import prepare_input_features
from src.predictor import (
    predict_customer_churn,
    analyze_feature_contributions,
    generate_business_recommendations,
    calculate_financial_impact
)

def test_prediction_output_structure(model, sample_customer_high_risk):
    df_encoded = prepare_input_features(sample_customer_high_risk)
    pred_label, churn_proba, retain_proba, risk_level, risk_color = predict_customer_churn(model, df_encoded)

    assert pred_label in ["Likely to Churn", "Likely to Stay", "Churn (Likely to leave)", "Retain (Likely to stay)"]
    assert 0.0 <= churn_proba <= 1.0
    assert 0.0 <= retain_proba <= 1.0
    assert round(churn_proba + retain_proba, 4) == 1.0
    assert risk_level in ["Low Risk", "Medium Risk", "High Risk"]
    assert risk_color.startswith("#")

def test_relative_risk_classification(model, sample_customer_high_risk, sample_customer_low_risk):
    df_high = prepare_input_features(sample_customer_high_risk)
    _, p_high, _, risk_high, _ = predict_customer_churn(model, df_high)

    df_low = prepare_input_features(sample_customer_low_risk)
    _, p_low, _, risk_low, _ = predict_customer_churn(model, df_low)

    assert p_high > p_low
    assert risk_high in ["Medium Risk", "High Risk"]
    assert risk_low == "Low Risk"

def test_analyze_feature_contributions(model, sample_customer_high_risk):
    df_encoded = prepare_input_features(sample_customer_high_risk)
    drivers = analyze_feature_contributions(model, df_encoded, top_n=5)

    assert isinstance(drivers, pd.DataFrame)
    assert len(drivers) <= 5
    assert "Feature" in drivers.columns
    assert "Contribution" in drivers.columns

def test_generate_business_recommendations(sample_customer_high_risk):
    recs = generate_business_recommendations(sample_customer_high_risk, churn_proba=0.75, risk_level="High Risk")
    assert isinstance(recs, list)
    assert len(recs) >= 1
    # Check that recommendation text contains actionable keywords
    combined_text = " ".join(recs).lower()
    assert any(term in combined_text for term in ["contract", "discount", "support", "offer", "check"])

def test_calculate_financial_impact(sample_customer_high_risk):
    impact = calculate_financial_impact(sample_customer_high_risk, churn_proba=0.8)
    assert isinstance(impact, dict)
    assert "clv" in impact
    assert "revenue_at_risk" in impact
    assert impact["clv"] > 0
    assert impact["revenue_at_risk"] > 0
