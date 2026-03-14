"""
Scenario B — 1D CNN multiclass classifier
Treats the 13 flow features as a 1D sequence and learns local feature
interactions via convolutional filters, then compares against RF baseline.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL

DATA_DIR = "data/scenario_b"
OUTPUT_DIR = "outputs/scenario_b"

BATCH_SIZE = 256
EPOCHS = 100       # max epochs
LR = 1e-3
PATIENCE = 5       # stop if val loss doesn't improve for 5 consecutive epochs


class TrafficCNN(nn.Module):
    """
    1D CNN for flow-based traffic classification.
    Input shape: (batch, 1, n_features) — features treated as a 1D signal.
    Two conv layers learn local feature relationships, FC layers classify.
    """
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
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
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


def prepare_data(data):
    X = data[FEATURE_COLS].values.astype(np.float32)
    y_raw = data[LABEL_COL].str.strip().values

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    return X_train, X_test, y_train, y_test, le, scaler


def make_loaders(X_train, X_test, y_train, y_test):
    # Add channel dim: (N, 13) -> (N, 1, 13)
    X_tr = torch.tensor(X_train).unsqueeze(1)
    X_te = torch.tensor(X_test).unsqueeze(1)
    y_tr = torch.tensor(y_train)
    y_te = torch.tensor(y_test)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH_SIZE)
    return train_loader, test_loader


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            if criterion is not None:
                total_loss += criterion(logits, y_batch).item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    val_loss = total_loss / len(loader) if criterion is not None else None
    return np.array(all_labels), np.array(all_preds), val_loss


def plot_training_curve(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("1D CNN Training vs Validation Loss — Scenario B")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


def plot_confusion_matrix(y_test, y_pred, le, out_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix — 1D CNN, Scenario B (14-class)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    data = load_data(DATA_DIR)
    data = clean_data(data)

    X_train, X_test, y_train, y_test, le, scaler = prepare_data(data)
    train_loader, test_loader = make_loaders(X_train, X_test, y_train, y_test)

    n_classes = len(le.classes_)
    model = TrafficCNN(n_features=len(FEATURE_COLS), n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining 1D CNN (max {EPOCHS} epochs, patience={PATIENCE})...")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_weights = None

    for epoch in range(EPOCHS):
        tr_loss = train(model, train_loader, optimizer, criterion)
        _, _, val_loss = evaluate(model, test_loader, criterion)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"  Epoch {epoch+1:3d} — Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break

    # Restore best weights before evaluating
    model.load_state_dict(best_weights)

    y_true, y_pred, _ = evaluate(model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    print(f"\nCNN Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    plot_training_curve(train_losses, val_losses, f"{OUTPUT_DIR}/cnn_training_loss.png")
    plot_confusion_matrix(y_true, y_pred, le, f"{OUTPUT_DIR}/confusion_matrix_cnn.png")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/cnn_model.pt")
    print("Model saved.")
