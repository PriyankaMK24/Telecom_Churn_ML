"""
preprocessing.py
----------------
Module to convert raw user input from Streamlit forms into the exact 30 encoded
feature columns expected by the trained Machine Learning model (churn_model.pkl).

Model Features (30 columns in exact order):
1. Tenure Months
2. Monthly Charges
3. Total Charges
4. Gender_Male
5. Senior Citizen_Yes
6. Partner_Yes
7. Dependents_Yes
8. Phone Service_Yes
9. Multiple Lines_No phone service
10. Multiple Lines_Yes
11. Internet Service_Fiber optic
12. Internet Service_No
13. Online Security_No internet service
14. Online Security_Yes
15. Online Backup_No internet service
16. Online Backup_Yes
17. Device Protection_No internet service
18. Device Protection_Yes
19. Tech Support_No internet service
20. Tech Support_Yes
21. Streaming TV_No internet service
22. Streaming TV_Yes
23. Streaming Movies_No internet service
24. Streaming Movies_Yes
25. Contract_One year
26. Contract_Two year
27. Paperless Billing_Yes
28. Payment Method_Credit card (automatic)
29. Payment Method_Electronic check
30. Payment Method_Mailed check
"""

import pandas as pd
import numpy as np

# Model feature order strictly required by model.feature_names_in_
FEATURE_ORDER = [
    'Tenure Months', 'Monthly Charges', 'Total Charges', 'Gender_Male',
    'Senior Citizen_Yes', 'Partner_Yes', 'Dependents_Yes', 'Phone Service_Yes',
    'Multiple Lines_No phone service', 'Multiple Lines_Yes',
    'Internet Service_Fiber optic', 'Internet Service_No',
    'Online Security_No internet service', 'Online Security_Yes',
    'Online Backup_No internet service', 'Online Backup_Yes',
    'Device Protection_No internet service', 'Device Protection_Yes',
    'Tech Support_No internet service', 'Tech Support_Yes',
    'Streaming TV_No internet service', 'Streaming TV_Yes',
    'Streaming Movies_No internet service', 'Streaming Movies_Yes',
    'Contract_One year', 'Contract_Two year', 'Paperless Billing_Yes',
    'Payment Method_Credit card (automatic)',
    'Payment Method_Electronic check', 'Payment Method_Mailed check'
]

def prepare_input_features(inputs: dict) -> pd.DataFrame:
    """
    Transforms a dictionary of raw customer inputs into a single-row DataFrame
    with 30 features ordered exactly as required by the trained model.

    Parameters:
        inputs (dict): Raw inputs captured from Streamlit UI form.

    Returns:
        pd.DataFrame: DataFrame (1 row, 30 columns) ready for model.predict()
    """

    # Extract numerical inputs
    tenure = float(inputs.get('tenure', 0))
    monthly_charges = float(inputs.get('monthly_charges', 0.0))
    total_charges = inputs.get('total_charges', None)
    
    # Calculate Total Charges automatically if not explicitly given or invalid
    if total_charges is None or float(total_charges) <= 0:
        total_charges = tenure * monthly_charges
    else:
        total_charges = float(total_charges)

    # Categorical logic mapping
    gender = inputs.get('gender', 'Female')
    senior = inputs.get('senior_citizen', 'No')
    partner = inputs.get('partner', 'No')
    dependents = inputs.get('dependents', 'No')
    phone = inputs.get('phone_service', 'Yes')
    multiple_lines = inputs.get('multiple_lines', 'No')
    internet = inputs.get('internet_service', 'DSL')
    online_sec = inputs.get('online_security', 'No')
    online_bkp = inputs.get('online_backup', 'No')
    dev_prot = inputs.get('device_protection', 'No')
    tech_sup = inputs.get('tech_support', 'No')
    stream_tv = inputs.get('streaming_tv', 'No')
    stream_mov = inputs.get('streaming_movies', 'No')
    contract = inputs.get('contract', 'Month-to-month')
    paperless = inputs.get('paperless_billing', 'Yes')
    payment = inputs.get('payment_method', 'Electronic check')

    # Helper for internet dependency override
    is_no_internet = (internet == 'No')
    is_no_phone = (phone == 'No')

    row_data = {
        'Tenure Months': tenure,
        'Monthly Charges': monthly_charges,
        'Total Charges': total_charges,
        
        # Demographics
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Senior Citizen_Yes': 1 if senior in [1, 'Yes', 'True', True] else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        
        # Phone Services
        'Phone Service_Yes': 1 if phone == 'Yes' else 0,
        'Multiple Lines_No phone service': 1 if is_no_phone else 0,
        'Multiple Lines_Yes': 1 if (phone == 'Yes' and multiple_lines == 'Yes') else 0,
        
        # Internet Services
        'Internet Service_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'Internet Service_No': 1 if is_no_internet else 0,
        
        # Internet Add-on Services
        'Online Security_No internet service': 1 if is_no_internet else 0,
        'Online Security_Yes': 1 if (not is_no_internet and online_sec == 'Yes') else 0,
        
        'Online Backup_No internet service': 1 if is_no_internet else 0,
        'Online Backup_Yes': 1 if (not is_no_internet and online_bkp == 'Yes') else 0,
        
        'Device Protection_No internet service': 1 if is_no_internet else 0,
        'Device Protection_Yes': 1 if (not is_no_internet and dev_prot == 'Yes') else 0,
        
        'Tech Support_No internet service': 1 if is_no_internet else 0,
        'Tech Support_Yes': 1 if (not is_no_internet and tech_sup == 'Yes') else 0,
        
        'Streaming TV_No internet service': 1 if is_no_internet else 0,
        'Streaming TV_Yes': 1 if (not is_no_internet and stream_tv == 'Yes') else 0,
        
        'Streaming Movies_No internet service': 1 if is_no_internet else 0,
        'Streaming Movies_Yes': 1 if (not is_no_internet and stream_mov == 'Yes') else 0,
        
        # Contract & Billing
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'Paperless Billing_Yes': 1 if paperless == 'Yes' else 0,
        
        # Payment Methods
        'Payment Method_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
        'Payment Method_Electronic check': 1 if payment == 'Electronic check' else 0,
        'Payment Method_Mailed check': 1 if payment == 'Mailed check' else 0
    }

    # Create single row dataframe with exact feature ordering
    df_encoded = pd.DataFrame([row_data])[FEATURE_ORDER]
    return df_encoded
