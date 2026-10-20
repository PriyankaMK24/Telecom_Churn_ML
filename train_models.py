"""
train_models.py
---------------
Model Training and Comparison Script for Telecom Customer Churn Prediction.
Builds an end-to-end scikit-learn Pipeline (ColumnTransformer + StandardScaler + Classifier),
fits strictly on training data (X_train), and evaluates on unseen test data (X_test).

Compares:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

Saves model evaluation metrics to Models/model_metrics.json and exports
the full fitted pipeline to Models/churn_model.pkl using Joblib.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
import numpy as np

# Define feature columns matching preprocessing pipeline (30 non-leakage columns)
FEATURE_COLS = [
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

NUMERICAL_COLS = ['Tenure Months', 'Monthly Charges', 'Total Charges']
CATEGORICAL_COLS = [col for col in FEATURE_COLS if col not in NUMERICAL_COLS]

def load_and_preprocess_data(data_path="Data/processed_data.csv"):
    """
    Loads raw/processed dataset and extracts feature matrix X and target y.
    Prevents data leakage by ensuring no target-derived features are in X.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Identify target column
    if 'Churn Value' in df.columns:
        y = df['Churn Value'].astype(int)
    elif 'Churn Label_Yes' in df.columns:
        y = df['Churn Label_Yes'].astype(int)
    elif 'Churn Label' in df.columns:
        y = df['Churn Label'].apply(lambda v: 1 if str(v).lower() in ['yes', '1', 'true'] else 0).astype(int)
    else:
        raise KeyError("Target column ('Churn Value' or 'Churn Label') not found in dataset.")

    # Select non-leakage features
    X = df[FEATURE_COLS].copy()

    # Convert boolean to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    # Handle missing values in numerical columns safely
    X['Total Charges'] = pd.to_numeric(X['Total Charges'], errors='coerce')
    X['Total Charges'] = X['Total Charges'].fillna(X['Tenure Months'] * X['Monthly Charges'])

    return X, y

def create_preprocessing_pipeline():
    """
    Creates a ColumnTransformer to scale numerical features and pass through binary indicators.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', 'passthrough', CATEGORICAL_COLS)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor

def train_and_evaluate():
    print("[+] Loading dataset...")
    X, y = load_and_preprocess_data()

    # 80% Train / 20% Test Split (Stratified to maintain churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[*] Dataset split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

    # Define base classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
    }

    results = {}
    fitted_pipelines = {}

    print("\n[*] Fitting preprocessing + model pipelines strictly on training data...")
    for name, clf in classifiers.items():
        preprocessor = create_preprocessing_pipeline()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Fit pipeline ONLY on X_train
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict ONLY on unseen X_test
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Compute ROC Curve coordinates (downsampled for clean UI rendering)
        fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
        idx_roc = np.linspace(0, len(fpr) - 1, num=min(30, len(fpr)), dtype=int)
        roc_data = {
            "fpr": [round(float(val), 4) for val in fpr[idx_roc]],
            "tpr": [round(float(val), 4) for val in tpr[idx_roc]]
        }

        # Compute Precision-Recall Curve coordinates
        pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, y_proba)
        idx_pr = np.linspace(0, len(pr_prec) - 1, num=min(30, len(pr_prec)), dtype=int)
        pr_data = {
            "precision": [round(float(val), 4) for val in pr_prec[idx_pr]],
            "recall": [round(float(val), 4) for val in pr_rec[idx_pr]]
        }

        # Find optimal probability threshold (maximizing F1 score)
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(pr_prec[:-1], pr_rec[:-1])]
        best_thresh_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
        best_threshold = float(pr_thresh[best_thresh_idx]) if len(pr_thresh) > 0 else 0.5

        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4),
            "ROC AUC": round(float(auc), 4),
            "Optimal Threshold": round(float(best_threshold), 4),
            "Confusion Matrix": cm,
            "ROC Curve": roc_data,
            "PR Curve": pr_data
        }

        print(f"  - {name:20s} | Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | Optimal Thresh: {best_threshold:.4f}")

    # Determine best model based on F1-Score & ROC-AUC
    best_model_name = max(results.keys(), key=lambda k: (results[k]["F1 Score"], results[k]["ROC AUC"]))
    print(f"\n[+] Best Performing Pipeline: {best_model_name} (F1={results[best_model_name]['F1 Score']}, AUC={results[best_model_name]['ROC AUC']})")

    # Save model evaluation metrics to Models/model_metrics.json
    os.makedirs("Models", exist_ok=True)
    with open("Models/model_metrics.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": results
        }, f, indent=4)

    print("[+] Saved evaluation metrics to Models/model_metrics.json")

    # Save Logistic Regression Pipeline for interpretability and coefficient log-odds analysis
    primary_pipeline = fitted_pipelines["Logistic Regression"]

    joblib.dump(primary_pipeline, "Models/churn_model.pkl")
    print("[+] Saved complete preprocessing + Logistic Regression pipeline to Models/churn_model.pkl")

    return results

if __name__ == "__main__":
    train_and_evaluate()



