# File: model_utils.py
# Purpose: Train, save, load, and evaluate digit classifier model

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

MODEL_PATH = "models/digit_model.pkl"

def train_model(csv_path="data/digit_data.csv"):
    """Train model from CSV dataset and save"""
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, acc, cm

def load_model():
    """Load trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first!")
    return joblib.load(MODEL_PATH)

def predict_single(model, features):
    """Predict single digit (numpy array of 64 features)"""
    # Wrap features in DataFrame with model's feature names if available
    if hasattr(model, "feature_names_in_"):
        features = pd.DataFrame([features], columns=model.feature_names_in_)
    else:
        features = np.array([features])
    pred = model.predict(features)[0]
    conf = np.max(model.predict_proba(features))
    return pred, conf

def predict_batch(model, X):
    """Predict batch of digits"""
    # Wrap features in DataFrame with model's feature names if available
    if hasattr(model, "feature_names_in_"):
        X = pd.DataFrame(X, columns=model.feature_names_in_)
    else:
        X = np.array(X)
    preds = model.predict(X)
    probs = np.max(model.predict_proba(X), axis=1)
    return preds, probs
