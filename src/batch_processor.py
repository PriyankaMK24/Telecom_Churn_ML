import pandas as pd
import numpy as np
from src.preprocessing import prepare_input_features
from src.predictor import predict_customer_churn, calculate_financial_impact

def process_batch_csv(df_raw: pd.DataFrame, model) -> tuple:
    """
    Processes a DataFrame of raw customer data, encodes features for each row,
    and appends ML predictions, churn probabilities, risk levels, CLV ($), and Revenue at Risk ($).

    Returns:
        tuple: (result_df, summary_dict)
    """
    column_mapping = {
        'tenure': 'tenure', 'Tenure Months': 'tenure', 'Tenure': 'tenure',
        'MonthlyCharges': 'monthly_charges', 'Monthly Charges': 'monthly_charges',
        'TotalCharges': 'total_charges', 'Total Charges': 'total_charges',
        'gender': 'gender', 'Gender': 'gender',
        'SeniorCitizen': 'senior_citizen', 'Senior Citizen': 'senior_citizen',
        'Partner': 'partner', 'Dependents': 'dependents',
        'PhoneService': 'phone_service', 'Phone Service': 'phone_service',
        'MultipleLines': 'multiple_lines', 'Multiple Lines': 'multiple_lines',
        'InternetService': 'internet_service', 'Internet Service': 'internet_service',
        'OnlineSecurity': 'online_security', 'Online Security': 'online_security',
        'OnlineBackup': 'online_backup', 'Online Backup': 'online_backup',
        'DeviceProtection': 'device_protection', 'Device Protection': 'device_protection',
        'TechSupport': 'tech_support', 'Tech Support': 'tech_support',
        'StreamingTV': 'streaming_tv', 'Streaming TV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies', 'Streaming Movies': 'streaming_movies',
        'Contract': 'contract', 'PaperlessBilling': 'paperless_billing', 'Paperless Billing': 'paperless_billing',
        'PaymentMethod': 'payment_method', 'Payment Method': 'payment_method'
    }

    df_clean = df_raw.copy()
    renamed_cols = {col: column_mapping[col] for col in df_clean.columns if col in column_mapping}
    df_clean.rename(columns=renamed_cols, inplace=True)

    predictions = []
    probabilities = []
    risk_levels = []
    clvs = []
    at_risks = []

    for idx, row in df_clean.iterrows():
        input_dict = row.to_dict()
        df_encoded = prepare_input_features(input_dict)
        pred_label, churn_proba, retain_proba, risk_level, _ = predict_customer_churn(model, df_encoded)
        fin = calculate_financial_impact(input_dict, churn_proba)

        predictions.append("Churn" if churn_proba >= 0.5 else "Retain")
        probabilities.append(round(churn_proba * 100, 2))
        risk_levels.append(risk_level)
        clvs.append(fin['clv'])
        at_risks.append(fin['revenue_at_risk'])

    result_df = df_raw.copy()
    result_df['Predicted_Status'] = predictions
    result_df['Churn_Probability_%'] = probabilities
    result_df['Risk_Level'] = risk_levels
    result_df['Estimated_CLV_$'] = clvs
    result_df['Revenue_At_Risk_$'] = at_risks

    total_records = len(result_df)
    high_risk_count = sum(1 for r in risk_levels if r == 'High Risk')
    churn_count = sum(1 for p in predictions if p == 'Churn')
    churn_rate = (churn_count / total_records * 100) if total_records > 0 else 0.0
    total_revenue_at_risk = round(sum(at_risks), 2)

    summary = {
        'total_records': total_records,
        'churn_rate': round(float(churn_rate), 1),
        'high_risk_count': high_risk_count,
        'total_revenue_at_risk_$': total_revenue_at_risk
    }

    return result_df, summary


