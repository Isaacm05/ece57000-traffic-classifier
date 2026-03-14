"""
Hierarchical Stage 1 — Binary VPN detector
Trained on Scenario A1. Classifies any flow as VPN or Non-VPN.
Returns the trained model + label encoder for use in pipeline.py.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL

DATA_DIR = "data/scenario_a1"
OUTPUT_DIR = "outputs/hierarchical"


def train_stage1():
    data = load_data(DATA_DIR)
    data = clean_data(data)

    X = data[FEATURE_COLS].values
    y_raw = data[LABEL_COL].str.strip().values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Stage 1 (VPN detector) accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, f"{OUTPUT_DIR}/stage1_model.pkl")
    joblib.dump(le, f"{OUTPUT_DIR}/stage1_le.pkl")
    print(f"Saved stage 1 model to {OUTPUT_DIR}/")

    return model, le


if __name__ == "__main__":
    train_stage1()
