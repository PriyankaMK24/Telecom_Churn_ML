"""
test_preprocessing.py
---------------------
Unit tests for data preprocessing and feature encoding logic.
"""

import pandas as pd
import numpy as np
from src.preprocessing import prepare_input_features, FEATURE_ORDER

def test_feature_count_and_order(sample_customer_high_risk):
    df_encoded = prepare_input_features(sample_customer_high_risk)
    assert isinstance(df_encoded, pd.DataFrame)
    assert df_encoded.shape == (1, 30)
    assert list(df_encoded.columns) == FEATURE_ORDER

def test_total_charges_imputation(sample_customer_high_risk):
    payload = sample_customer_high_risk.copy()
    payload["total_charges"] = None
    payload["tenure"] = 10
    payload["monthly_charges"] = 50.0

    df_encoded = prepare_input_features(payload)
    # Total charges should be tenure * monthly_charges = 500.0
    assert df_encoded["Total Charges"].iloc[0] == 500.0

def test_contract_encoding(sample_customer_low_risk):
    df_encoded = prepare_input_features(sample_customer_low_risk)
    assert df_encoded["Contract_Two year"].iloc[0] == 1
    assert df_encoded["Contract_One year"].iloc[0] == 0

def test_internet_service_fiber_optic(sample_customer_high_risk):
    df_encoded = prepare_input_features(sample_customer_high_risk)
    assert df_encoded["Internet Service_Fiber optic"].iloc[0] == 1
    assert df_encoded["Internet Service_No"].iloc[0] == 0

def test_no_internet_service_cascading(sample_customer_low_risk):
    df_encoded = prepare_input_features(sample_customer_low_risk)
    assert df_encoded["Internet Service_No"].iloc[0] == 1
    assert df_encoded["Online Security_No internet service"].iloc[0] == 1
    assert df_encoded["Tech Support_No internet service"].iloc[0] == 1
