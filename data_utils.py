"""
Shared data loading and cleaning utilities for all experiments.
Same ARFF parser as traffic_classifier.py, but scenario-agnostic.
"""
import os
import glob
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "duration",
    "total_fiat",
    "total_biat",
    "min_fiat",
    "min_biat",
    "max_fiat",
    "max_biat",
    "mean_fiat",
    "mean_biat",
    "flowPktsPerSecond",
    "flowBytesPerSecond",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
    "min_active",
    "mean_active",
    "max_active",
    "std_active",
    "min_idle",
    "mean_idle",
    "max_idle",
    "std_idle",
]

LABEL_COL = "class1"


def load_data(data_dir):
    arff_files = glob.glob(os.path.join(data_dir, "*.arff"))
    if not arff_files:
        raise FileNotFoundError(f"No ARFF files found in {data_dir}")
    print(f"Found {len(arff_files)} file(s) in {data_dir}, loading...")

    frames = []
    for file in arff_files:
        with open(file, "r") as f:
            lines = f.readlines()
        data_start = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("@DATA"))
        header_lines = lines[:data_start]
        data_lines = lines[data_start + 1:]
        cols = []
        for line in header_lines:
            if line.strip().upper().startswith("@ATTRIBUTE"):
                cols.append(line.strip().split()[1])
        rows = [l.strip().split(",") for l in data_lines if l.strip()]
        df = pd.DataFrame(rows, columns=cols)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(data)}")
    print(f"Classes found: {data[LABEL_COL].unique()}")
    return data


def clean_data(data):
    data = data[FEATURE_COLS + [LABEL_COL]].copy()
    data = data[data[LABEL_COL].str.strip() != ""]
    for col in FEATURE_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    before = len(data)
    data = data.dropna()
    print(f"Rows after cleaning: {len(data)} (removed {before - len(data)})")
    return data
