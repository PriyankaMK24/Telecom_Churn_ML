import os
import joblib
import pandas as pd
import numpy as np
import warnings
from src.preprocessing import FEATURE_ORDER, prepare_input_features

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def load_churn_model(model_path: str = "Models/churn_model.pkl"):
    """
    Loads the trained Machine Learning pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")

    model = joblib.load(model_path)
    return model

def predict_customer_churn(model, df_encoded: pd.DataFrame):
    """
    Predicts churn class and probability score for a given encoded customer DataFrame.

    Returns:
        tuple: (prediction_label, churn_probability, retain_probability, risk_level, risk_color)
    """
    expected_num_features = 30
    if hasattr(model, "feature_names_in_"):
        expected_num_features = len(model.feature_names_in_)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get('classifier'), "feature_names_in_"):
        expected_num_features = len(model.named_steps['classifier'].feature_names_in_)

    if df_encoded.shape[1] != expected_num_features:
        raise ValueError(
            f"Feature size mismatch! Model expects {expected_num_features} features, "
            f"but got {df_encoded.shape[1]}."
        )

    # Prediction using Pipeline / Estimator
    prediction = int(model.predict(df_encoded)[0])
    probabilities = model.predict_proba(df_encoded)[0]

    churn_proba = float(probabilities[1])
    retain_proba = float(probabilities[0])

    # Classify Risk Level:
    # 🟢 Low: 0–30% | 🟡 Medium: 31–60% | 🔴 High: 61–100%
    if churn_proba <= 0.30:
        risk_level = "Low Risk"
        risk_color = "#22c55e"  # Vibrant Green
    elif churn_proba <= 0.60:
        risk_level = "Medium Risk"
        risk_color = "#f59e0b"  # Warm Gold/Amber
    else:
        risk_level = "High Risk"
        risk_color = "#ef4444"  # Vibrant Red

    pred_label = "Churn (Likely to leave)" if prediction == 1 else "Retain (Likely to stay)"

    return pred_label, churn_proba, retain_proba, risk_level, risk_color

def calculate_financial_impact(inputs: dict, churn_proba: float) -> dict:
    """
    Calculates Customer Lifetime Value (CLV), Revenue at Risk, and Max Recommended Retention Budget.
    """
    tenure = float(inputs.get('tenure', 12))
    monthly = float(inputs.get('monthly_charges', 70.0))

    # Assume 24-month horizon or remaining subscription value
    expected_months = max(12.0, 36.0 - min(tenure, 24.0))
    clv = round(monthly * expected_months, 2)
    revenue_at_risk = round(clv * churn_proba, 2)
    recommended_budget = round(revenue_at_risk * 0.15, 2)  # 15% max retention offer cap

    return {
        "clv": clv,
        "monthly_charges": monthly,
        "revenue_at_risk": revenue_at_risk,
        "recommended_retention_budget": recommended_budget,
        "horizon_months": int(expected_months)
    }

def analyze_feature_contributions(model, df_encoded: pd.DataFrame, top_n: int = 5):
    """
    Calculates log-odds contribution (Coefficient * Feature Value) for each feature.
    Handles scikit-learn Pipeline objects seamlessly by applying scaling transform first.

    Returns:
        pd.DataFrame: Top N drivers with feature name, coefficient, value, and log-odds contribution.
    """
    if hasattr(model, "named_steps"):
        classifier = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        
        # Transform features through ColumnTransformer/Scaler
        transformed_val = preprocessor.transform(df_encoded)
        if hasattr(transformed_val, "toarray"):
            transformed_val = transformed_val.toarray()
        
        values = transformed_val[0]
        coefs = classifier.coef_[0] if hasattr(classifier, "coef_") else np.zeros(len(values))
        
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = FEATURE_ORDER
    else:
        coefs = model.coef_[0] if hasattr(model, "coef_") else np.zeros(df_encoded.shape[1])
        feature_names = getattr(model, "feature_names_in_", FEATURE_ORDER)
        values = df_encoded.iloc[0].values

    contributions = coefs * values

    df_contrib = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Value': df_encoded.iloc[0].values,
        'Contribution': contributions
    })

    # Sort by positive contribution towards churn risk
    churn_drivers = df_contrib.sort_values(by='Contribution', ascending=False).head(top_n)
    return churn_drivers

def compute_shap_contributions(model, df_encoded: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Computes SHAP attribution values for exact feature influence breakdown.
    Falls back gracefully to log-odds calculation if SHAP is not installed or fails.
    """
    if not HAS_SHAP:
        return analyze_feature_contributions(model, df_encoded, top_n=top_n)

    try:
        if hasattr(model, "named_steps"):
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            transformed_val = preprocessor.transform(df_encoded)
            if hasattr(transformed_val, "toarray"):
                transformed_val = transformed_val.toarray()
            
            explainer = shap.LinearExplainer(classifier, transformed_val)
            shap_values = explainer.shap_values(transformed_val)
            
            if isinstance(shap_values, list):
                shap_arr = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                shap_arr = shap_values[0] if shap_values.ndim > 1 else shap_values

            feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, "get_feature_names_out") else FEATURE_ORDER
        else:
            explainer = shap.Explainer(model, df_encoded)
            shap_res = explainer(df_encoded)
            shap_arr = shap_res.values[0]
            feature_names = df_encoded.columns.tolist()

        df_shap = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_arr,
            'Absolute_Impact': np.abs(shap_arr)
        }).sort_values(by='Absolute_Impact', ascending=False).head(top_n)

        return df_shap
    except Exception as e:
        warnings.warn(f"SHAP computation warning: {e}. Falling back to log-odds driver analysis.")
        return analyze_feature_contributions(model, df_encoded, top_n=top_n)

def simulate_scenario_impact(model, baseline_inputs: dict, scenario_overrides: dict) -> dict:
    """
    Simulates what-if scenario (e.g., upgrading contract, offering monthly discount, adding tech support).
    Compares baseline vs modified churn risk & financial impact.
    """
    modified_inputs = baseline_inputs.copy()
    modified_inputs.update(scenario_overrides)

    df_base = prepare_input_features(baseline_inputs)
    df_mod = prepare_input_features(modified_inputs)

    _, base_proba, _, base_risk, base_color = predict_customer_churn(model, df_base)
    _, mod_proba, _, mod_risk, mod_color = predict_customer_churn(model, df_mod)

    base_fin = calculate_financial_impact(baseline_inputs, base_proba)
    mod_fin = calculate_financial_impact(modified_inputs, mod_proba)

    prob_delta = round((mod_proba - base_proba) * 100, 1)  # negative means risk reduced
    revenue_saved = round(base_fin['revenue_at_risk'] - mod_fin['revenue_at_risk'], 2)

    return {
        "baseline": {
            "churn_probability": round(base_proba * 100, 1),
            "risk_level": base_risk,
            "risk_color": base_color,
            "financial": base_fin
        },
        "simulated": {
            "churn_probability": round(mod_proba * 100, 1),
            "risk_level": mod_risk,
            "risk_color": mod_color,
            "financial": mod_fin
        },
        "impact": {
            "probability_change_%": prob_delta,
            "revenue_saved_$": max(0.0, revenue_saved),
            "risk_reduced": base_proba > mod_proba
        }
    }

def generate_business_recommendations(inputs: dict, churn_proba: float, risk_level: str) -> list:
    """
    Generates actionable, data-driven business recommendations based on customer attributes
    and churn probability score.
    """
    recs = []

    contract = inputs.get('contract', 'Month-to-month')
    payment = inputs.get('payment_method', '')
    internet = inputs.get('internet_service', '')
    tech_sup = inputs.get('tech_support', 'No')
    online_sec = inputs.get('online_security', 'No')
    tenure = float(inputs.get('tenure', 0))
    monthly = float(inputs.get('monthly_charges', 0.0))

    if risk_level == "High Risk":
        recs.append("🚨 **Immediate Retention Trigger**: Assign a dedicated Customer Success agent to contact this customer within 24 hours with a personalized retention offer.")
    elif risk_level == "Medium Risk":
        recs.append("⚠️ **Proactive Engagement**: Target this customer with promotional retention campaigns and plan optimization before contract renewal.")
    else:
        recs.append("✅ **Loyalty Program Candidate**: Customer is stable. Consider upselling premium services or introducing loyalty rewards.")

    # Contract specific recommendation
    if contract == 'Month-to-month':
        recs.append("📜 **Contract Upgrade Offer**: Month-to-month contracts have 4x higher churn risk. Offer a **15% annual discount** for switching to a 1-year or 2-year contract.")

    # Payment method recommendation
    if payment == 'Electronic check':
        recs.append("💳 **Automated Payment Incentive**: Electronic check users have high churn correlation. Offer a **$10 one-time bill credit** to enroll in Auto-Pay via Credit Card or Bank Transfer.")

    # Internet & Add-on service recommendations
    if internet == 'Fiber optic' and tech_sup == 'No':
        recs.append("🛠️ **Bundled Tech Support**: Fiber optic customers without Tech Support experience higher service friction. Offer **3 months of complimentary Tech Support**.")

    if internet != 'No' and online_sec == 'No':
        recs.append("🔒 **Security Bundle Promotion**: Add Online Security package at a discounted rate to increase service stickiness.")

    # Tenure recommendation
    if tenure < 12:
        recs.append("🎁 **New Customer Onboarding**: Customer is in the critical first year. Send targeted onboarding guides and usage satisfaction surveys.")

    # High Monthly Charges recommendation
    if monthly > 80.0:
        recs.append(f"💰 **Bill Optimization Review**: High monthly bill (${monthly:.2f}) detected. Conduct a plan review to optimize features and offer tailored discounts.")

    return recs


