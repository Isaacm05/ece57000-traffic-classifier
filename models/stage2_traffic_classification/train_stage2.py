"""
Hierarchical Stage 2 — Traffic type classifiers
Trains two separate 1D CNNs on Scenario B:
  - One on VPN-only flows (VPN-VOIP, VPN-CHAT, VPN-FT, etc.)
  - One on non-VPN flows (VOIP, CHAT, FT, etc.)
Each classifier only needs to distinguish 7 classes instead of 14.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL

DATA_DIR = "data/scenario_b"
OUTPUT_DIR = "outputs/hierarchical"

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
PATIENCE = 5


class TrafficCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for X_b, y_b in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    preds, labels, total = [], [], 0
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b)
            total += criterion(logits, y_b).item()
            preds.extend(logits.argmax(1).numpy())
            labels.extend(y_b.numpy())
    return np.array(labels), np.array(preds), total / len(loader)


def train_cnn(X_train, X_test, y_train, y_test, n_classes, label):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float32))
    X_test = scaler.transform(X_test.astype(np.float32))

    X_tr = torch.tensor(X_train).unsqueeze(1)
    X_te = torch.tensor(X_test).unsqueeze(1)
    y_tr = torch.tensor(y_train.astype(np.int64))
    y_te = torch.tensor(y_test.astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH_SIZE)

    model = TrafficCNN(n_features=len(FEATURE_COLS), n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val, epochs_no_improve, best_weights = float("inf"), 0, None
    print(f"\nTraining Stage 2 CNN [{label}] (max {EPOCHS} epochs, patience={PATIENCE})...")

    for epoch in range(EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        _, _, val_loss = eval_epoch(model, test_loader, criterion)
        print(f"  Epoch {epoch+1:3d} — Train: {tr_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_weights)
    y_true, y_pred, _ = eval_epoch(model, test_loader, criterion)
    acc = accuracy_score(y_true, y_pred)
    print(f"{label} accuracy: {acc:.4f}")

    return model, scaler, acc, y_true, y_pred


def train_stage2(train_data=None, test_data=None):
    if train_data is None or test_data is None:
        import pandas as pd
        from sklearn.model_selection import train_test_split as tts
        data = load_data(DATA_DIR)
        data = clean_data(data)
        data["class1"] = data["class1"].str.strip()
        train_data, test_data = tts(data, test_size=0.2, random_state=42, stratify=data["class1"])
    else:
        train_data = train_data.copy()
        test_data = test_data.copy()
        train_data["class1"] = train_data["class1"].str.strip()
        test_data["class1"] = test_data["class1"].str.strip()

    # Split into VPN and non-VPN subsets
    vpn_mask = train_data["class1"].str.startswith("VPN-")
    vpn_data = train_data[vpn_mask].copy()
    nonvpn_data = train_data[~vpn_mask].copy()

    vpn_test = test_data[test_data["class1"].str.startswith("VPN-")].copy()
    nonvpn_test = test_data[~test_data["class1"].str.startswith("VPN-")].copy()

    # Strip "VPN-" prefix so labels are comparable (VPN-VOIP -> VOIP)
    vpn_data["class1"] = vpn_data["class1"].str.replace("VPN-", "", regex=False)
    vpn_test["class1"] = vpn_test["class1"].str.replace("VPN-", "", regex=False)

    print(f"\nVPN train: {len(vpn_data)} | VPN test: {len(vpn_test)}")
    print(f"NonVPN train: {len(nonvpn_data)} | NonVPN test: {len(nonvpn_test)}")

    results = {}
    for label, train_subset, test_subset in [
        ("VPN", vpn_data, vpn_test),
        ("NonVPN", nonvpn_data, nonvpn_test),
    ]:
        le = LabelEncoder()
        y_train = le.fit_transform(train_subset["class1"].values)
        y_test = le.transform(test_subset["class1"].values)
        X_train = train_subset[FEATURE_COLS].values
        X_test = test_subset[FEATURE_COLS].values

        model, scaler, acc, y_true, y_pred = train_cnn(
            X_train, X_test, y_train, y_test, n_classes=len(le.classes_), label=label
        )
        print(classification_report(y_true, y_pred, target_names=le.classes_))

        torch.save(model.state_dict(), f"{OUTPUT_DIR}/stage2_{label.lower()}_cnn.pt")
        joblib.dump(le, f"{OUTPUT_DIR}/stage2_{label.lower()}_le.pkl")
        joblib.dump(scaler, f"{OUTPUT_DIR}/stage2_{label.lower()}_scaler.pkl")
        results[label] = {"acc": acc, "model": model, "le": le, "scaler": scaler}

    return results


if __name__ == "__main__":
    train_stage2()
