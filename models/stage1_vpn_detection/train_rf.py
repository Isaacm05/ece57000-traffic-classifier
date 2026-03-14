import os
import glob
import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
FEATURE_COLS = [
    "duration",
    "total_fiat",
    "total_biat",
    "min_fiat",
    "max_fiat",
    "mean_fiat",
    "mean_biat",
    "flowPktsPerSecond",
    "flowBytesPerSecond",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
]

LABEL_COL = "class1"

# Grab csv files, but its arff unfortunately
def load_data(data_dir):
    arff_files = glob.glob(os.path.join(data_dir, "*.arff"))

    if not arff_files:
        raise FileNotFoundError(f"No ARFF files found in {data_dir}")

    print(f"Found {len(arff_files)} file(s), loading...")

    frames = []
    for file in arff_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        # Find where data starts
        data_start = next(i for i, l in enumerate(lines) if l.strip().upper().startswith('@DATA'))
        header_lines = lines[:data_start]
        data_lines = lines[data_start+1:]
        
        # Extract column names
        cols = []
        for line in header_lines:
            if line.strip().upper().startswith('@ATTRIBUTE'):
                cols.append(line.strip().split()[1])
        
        # Parse data rows, skip empty lines
        rows = [l.strip().split(',') for l in data_lines if l.strip()]
        df = pd.DataFrame(rows, columns=cols)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(data)}")
    print(f"Classes found: {data[LABEL_COL].unique()}")
    return data


# Get the classes i want, replace infinity with NaN, drop rows with NaN values, return
def clean_data(data):
    data = data[FEATURE_COLS + [LABEL_COL]].copy()
    data = data[data[LABEL_COL].str.strip() != ""]
    data = data.replace([np.inf, -np.inf], np.nan)
    before = len(data)
    data = data.dropna()
    print(f"Rows after cleaning: {len(data)} (removed {before - len(data)})")
    return data


def train_model(data):
    X = data[FEATURE_COLS].values
    y_raw = data[LABEL_COL].values

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return model, label_encoder, X_test, y_test, y_pred


def plot_confusion_matrix(y_test, y_pred, le):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("Confusion Matrix - Random Forest Baseline")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    data = load_data("data/scenario_a1")
    data = clean_data(data)
    model, le, X_test, y_test, y_pred = train_model(data)
    plot_confusion_matrix(y_test, y_pred, le)