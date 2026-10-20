"""Train three leakage-safe churn classifiers and save a reusable artifact."""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "processed_data.csv"
MODEL_DIR = ROOT / "Models"
ARTIFACT_PATH = MODEL_DIR / "churn_models.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
TARGET_CANDIDATES = ("churn value", "churn", "churn label", "exited", "is churn")


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).strip().lower()).strip()


def find_target(columns) -> str:
    by_normalized = {normalized(c): c for c in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in by_normalized:
            return by_normalized[candidate]
    candidates = [c for c in columns if "churn" in normalized(c)]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Could not identify one target column. Expected a column such as 'Churn Value' or 'Churn Label'.")


def target_to_binary(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("The target column has missing values. Please remove or fill those rows in the source CSV.")
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        unique = set(pd.unique(values))
        if unique <= {0, 1, False, True} and len(unique) == 2:
            return values.astype(int)
    mapping = {"yes": 1, "true": 1, "churn": 1, "churned": 1, "1": 1,
               "no": 0, "false": 0, "stay": 0, "stayed": 0, "not churned": 0, "0": 0}
    mapped = values.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any() or mapped.nunique() != 2:
        raise ValueError(f"Target values must represent two churn classes. Found: {values.dropna().unique()[:10].tolist()}")
    return mapped.astype(int)


def load_dataset(path: Path = DATA_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Place your processed CSV there or update DATA_PATH in train_model.py.")
    df = pd.read_csv(path)
    target = find_target(df.columns)
    y = target_to_binary(df[target])
    drop_cols = []
    for col in df.columns:
        key = normalized(col)
        if col == target:
            drop_cols.append(col)
        elif ("churn" in key or "target" in key or key in {"exited", "is churn"}
              or key in {"id", "customer id", "customerid", "row id", "index"}
              or key.startswith("unnamed ")):
            drop_cols.append(col)
    X = df.drop(columns=drop_cols).copy()
    # Discard columns that contain no information and rows with no usable inputs.
    X = X.loc[:, X.nunique(dropna=True) > 1]
    valid = X.notna().any(axis=1)
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if X.empty:
        raise ValueError("No usable feature columns remain after removing target-derived and identifier columns.")
    return df.loc[valid].reset_index(drop=True), X, y, target, drop_cols


def make_pipeline(X: pd.DataFrame, estimator):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric),
                              ("categorical", categorical_pipe, categorical)], remainder="drop")
    return Pipeline([("preprocess", prep), ("classifier", estimator)])


def score_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def train_model(data_path: Path = DATA_PATH):
    original, X, y, target, dropped = load_dataset(data_path)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Need at least two examples in each churn class for a stratified train-test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    fitted, metrics, roc_curves = {}, {}, {}
    for name, estimator in models.items():
        pipeline = make_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = score_probability(pipeline, X_test)
        if proba is not None and y_test.nunique() == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(y_test, proba)
            roc_curves[name] = {
                "false_positive_rate": false_positive_rate.tolist(),
                "true_positive_rate": true_positive_rate.tolist(),
            }
        metrics[name] = {
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1-score": float(f1_score(y_test, pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, proba)) if proba is not None and y_test.nunique() == 2 else None,
            "Confusion matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        }
        fitted[name] = pipeline
        print(f"{name}: F1={metrics[name]['F1-score']:.3f}, recall={metrics[name]['Recall']:.3f}, accuracy={metrics[name]['Accuracy']:.3f}")
    best = max(metrics, key=lambda name: (metrics[name]["F1-score"], metrics[name]["ROC-AUC"] or 0))
    # Permutation importance on the held-out set is for reporting only; it does not fit or select the model.
    from sklearn.inspection import permutation_importance
    importance = permutation_importance(fitted[best], X_test, y_test, n_repeats=5,
                                        random_state=42, scoring="f1", n_jobs=-1)
    importance_rows = sorted(
        [(str(feature), float(score)) for feature, score in zip(X.columns, importance.importances_mean)],
        key=lambda item: item[1], reverse=True)
    artifact = {
        "models": fitted, "best_model": best, "feature_columns": X.columns.tolist(),
        "target_column": target, "target_positive_value": 1,
        "metrics": metrics, "roc_curves": roc_curves, "feature_importance": importance_rows,
        "dataset_path": str(data_path), "dataset_rows": int(len(original)),
        "dataset_columns": original.columns.tolist(), "excluded_columns": dropped,
        "positive_count": int((y == 1).sum()), "negative_count": int((y == 0).sum()),
        "test_size": 0.2, "random_state": 42,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump({"best_model": best, "metrics": metrics, "roc_curves": roc_curves, "target_column": target,
                   "feature_columns": X.columns.tolist(), "excluded_columns": dropped}, file, indent=2)
    print(f"Best model by held-out F1-score: {best}")
    print(f"Saved model artifact: {ARTIFACT_PATH}")
    return artifact


if __name__ == "__main__":
    train_model()
