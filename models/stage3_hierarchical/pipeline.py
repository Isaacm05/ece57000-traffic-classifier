"""
Hierarchical Pipeline — full end-to-end evaluation
Runs all three stages and prints a final comparison table:
  Flat RF vs Flat CNN vs Hierarchical (Stage1 RF + Stage2 CNN)
"""
import os, sys
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sklearn.model_selection import train_test_split
from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL
from models.stage1_vpn_detection.train_and_save import train_stage1
from models.stage2_traffic_classification.train_stage2 import train_stage2, TrafficCNN

OUTPUT_DIR = "outputs/hierarchical"
DATA_DIR_B = "data/scenario_b"


def load_or_train_stage1():
    model_path = f"{OUTPUT_DIR}/stage1_model.pkl"
    le_path = f"{OUTPUT_DIR}/stage1_le.pkl"
    if os.path.exists(model_path) and os.path.exists(le_path):
        print("Loading saved Stage 1 model...")
        return joblib.load(model_path), joblib.load(le_path)
    print("Training Stage 1 model...")
    return train_stage1()


def run_hierarchical_pipeline(data, stage1_model, stage1_le, stage2_results):
    """
    For each test flow:
      1. Stage 1 predicts VPN or Non-VPN
      2. Route to the matching Stage 2 CNN for traffic type
      3. Combine into a final 14-class prediction
    """
    data = data.copy()
    data["class1"] = data["class1"].str.strip()

    X_all = data[FEATURE_COLS].values.astype(np.float32)
    y_true_raw = data["class1"].values

    # Stage 1 — predict VPN vs Non-VPN
    stage1_preds = stage1_le.inverse_transform(stage1_model.predict(X_all))
    is_vpn = stage1_preds == "VPN"

    final_preds = np.empty(len(data), dtype=object)

    for label, mask in [("VPN", is_vpn), ("NonVPN", ~is_vpn)]:
        if mask.sum() == 0:
            continue
        X_sub = X_all[mask]
        model_info = stage2_results[label]
        le: LabelEncoder = model_info["le"]
        scaler: StandardScaler = model_info["scaler"]
        model: TrafficCNN = model_info["model"]

        X_scaled = scaler.transform(X_sub)
        X_tensor = torch.tensor(X_scaled).unsqueeze(1)

        model.eval()
        with torch.no_grad():
            preds = model(X_tensor).argmax(1).numpy()

        type_labels = le.inverse_transform(preds)
        # Re-attach VPN- prefix for VPN subset
        if label == "VPN":
            type_labels = np.array([f"VPN-{t}" for t in type_labels])
        final_preds[mask] = type_labels

    return y_true_raw, final_preds


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Load Scenario B data and do ONE shared split for fair comparison
    data = load_data(DATA_DIR_B)
    data = clean_data(data)
    data["class1"] = data["class1"].str.strip()

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["class1"]
    )
    print(f"Shared split — Train: {len(train_data)} | Test: {len(test_data)}")

    # Train/load all stages using the same train split
    stage1_model, stage1_le = load_or_train_stage1()
    stage2_results = train_stage2(train_data=train_data, test_data=test_data)

    # Evaluate hierarchical pipeline on the shared test set only
    y_true, y_pred = run_hierarchical_pipeline(test_data, stage1_model, stage1_le, stage2_results)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"Hierarchical Pipeline Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred))

    plot_confusion_matrix(
        y_true, y_pred,
        "Confusion Matrix — Hierarchical Pipeline (Stage1 RF + Stage2 CNN)",
        f"{OUTPUT_DIR}/confusion_matrix_hierarchical.png"
    )

    # Summary comparison table
    print("\n" + "="*50)
    print("MODEL COMPARISON — Scenario B (14-class)")
    print("="*50)
    print(f"{'Model':<30} {'Accuracy':>10}")
    print(f"{'-'*40}")
    print(f"{'Flat RF (13 features)':<30} {'0.5410':>10}")
    print(f"{'Flat CNN (13 features)':<30} {'0.6004':>10}")
    print(f"{'Hierarchical CNN (23 features)':<30} {acc:>10.4f}")
    print("="*50)
