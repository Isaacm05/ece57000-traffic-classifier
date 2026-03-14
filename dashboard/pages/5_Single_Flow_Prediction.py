import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL
from models.stage2_traffic_classification.train_stage2 import TrafficCNN

st.set_page_config(page_title="Single Flow Prediction", layout="wide")
st.title("Single Flow Prediction")
st.markdown("Sample a real flow from the dataset or enter your own values to get a live hierarchical prediction.")

@st.cache_data
def get_data():
    data = load_data("data/scenario_b")
    data = clean_data(data)
    data["class1"] = data["class1"].str.strip()
    return data

@st.cache_resource
def get_stage1():
    return joblib.load("outputs/hierarchical/stage1_model.pkl"), joblib.load("outputs/hierarchical/stage1_le.pkl")

@st.cache_resource
def get_stage2():
    models = {}
    for label in ["vpn", "nonvpn"]:
        le     = joblib.load(f"outputs/hierarchical/stage2_{label}_le.pkl")
        scaler = joblib.load(f"outputs/hierarchical/stage2_{label}_scaler.pkl")
        cnn    = TrafficCNN(n_features=len(FEATURE_COLS), n_classes=len(le.classes_))
        cnn.load_state_dict(torch.load(f"outputs/hierarchical/stage2_{label}_cnn.pt", weights_only=True))
        cnn.eval()
        models[label] = {"model": cnn, "le": le, "scaler": scaler}
    return models

data = get_data()
stage1_model, stage1_le = get_stage1()
stage2 = get_stage2()

# ── Initialize session state with first row of default class ─────────────────
if f"feat_{FEATURE_COLS[0]}" not in st.session_state:
    default_row = data.iloc[0]
    for feat in FEATURE_COLS:
        st.session_state[f"feat_{feat}"] = float(default_row[feat])
    st.session_state["true_label"] = default_row["class1"]

# ── Sample selector ───────────────────────────────────────────────────────────
st.subheader("Load a Sample Flow")
col_sel, col_btn = st.columns([3, 1])
sample_class = col_sel.selectbox("Traffic type to sample", sorted(data["class1"].unique()))
if col_btn.button("Load Sample", type="secondary"):
    row = data[data["class1"] == sample_class].sample(1, random_state=np.random.randint(0, 1000))
    st.session_state["true_label"] = row["class1"].values[0]
    for feat in FEATURE_COLS:
        st.session_state[f"feat_{feat}"] = float(row[feat].values[0])
    st.rerun()

# ── Feature inputs ────────────────────────────────────────────────────────────
st.subheader("Flow Features")
st.caption("Values auto-filled from sampled flow. Edit any value to test manually.")
cols = st.columns(4)
feature_values = []
for i, feat in enumerate(FEATURE_COLS):
    val = cols[i % 4].number_input(feat, format="%.2f", key=f"feat_{feat}")
    feature_values.append(val)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Classify This Flow", type="primary"):
    x = np.array(feature_values, dtype=np.float32).reshape(1, -1)

    # Stage 1
    vpn_label = stage1_le.inverse_transform(stage1_model.predict(x))[0]
    vpn_proba = stage1_model.predict_proba(x).max()

    # Stage 2
    key = "vpn" if vpn_label == "VPN" else "nonvpn"
    info = stage2[key]
    x_sc = info["scaler"].transform(x)
    x_t  = torch.tensor(x_sc).unsqueeze(1)
    with torch.no_grad():
        logits = info["model"](x_t)
        probs  = torch.softmax(logits, dim=1).numpy()[0]
    pred_idx   = probs.argmax()
    type_label = info["le"].inverse_transform([pred_idx])[0]
    type_conf  = probs[pred_idx]
    final = f"VPN-{type_label}" if vpn_label == "VPN" else type_label

    st.divider()
    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage 1 — VPN?", vpn_label, f"{vpn_proba:.1%} confidence")
    c2.metric("Stage 2 — Type?", type_label, f"{type_conf:.1%} confidence")
    c3.metric("Final Prediction", final)
    true_label = st.session_state.get("true_label", "unknown")
    c4.metric("True Label", true_label,
              "✓ Correct" if final == true_label else "✗ Wrong")

    st.subheader(f"Stage 2 Class Probabilities ({vpn_label} branch)")
    prob_df = pd.Series(dict(zip(info["le"].classes_, probs))).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#2ecc71" if c == type_label else "#9b59b6" for c in prob_df.index]
    ax.bar(prob_df.index, prob_df.values, color=colors)
    ax.set_ylabel("Probability"); ax.set_ylim(0, 1)
    ax.set_title("Green = predicted class")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    st.pyplot(fig)
