import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL
from models.stage2_traffic_classification.train_stage2 import TrafficCNN

st.set_page_config(page_title="Network Traffic Classifier", page_icon="🔍", layout="wide")

# ── Load everything ───────────────────────────────────────────────────────────
@st.cache_data
def get_data_b():
    data = load_data("data/scenario_b")
    data = clean_data(data)
    data["class1"] = data["class1"].str.strip()
    return data

@st.cache_resource
def get_models():
    s1_model = joblib.load("outputs/hierarchical/stage1_model.pkl")
    s1_le    = joblib.load("outputs/hierarchical/stage1_le.pkl")
    s2 = {}
    for label in ["vpn", "nonvpn"]:
        le     = joblib.load(f"outputs/hierarchical/stage2_{label}_le.pkl")
        scaler = joblib.load(f"outputs/hierarchical/stage2_{label}_scaler.pkl")
        cnn    = TrafficCNN(n_features=len(FEATURE_COLS), n_classes=len(le.classes_))
        cnn.load_state_dict(torch.load(f"outputs/hierarchical/stage2_{label}_cnn.pt", weights_only=True))
        cnn.eval()
        s2[label] = {"model": cnn, "le": le, "scaler": scaler}
    return s1_model, s1_le, s2

@st.cache_data
def run_pipeline(_s1_model, _s1_le, _s2):
    data = get_data_b()
    _, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["class1"])
    X_all = test[FEATURE_COLS].values.astype(np.float32)
    stage1_preds = _s1_le.inverse_transform(_s1_model.predict(X_all))
    is_vpn = stage1_preds == "VPN"
    final_preds = np.empty(len(test), dtype=object)
    for key, mask in [("vpn", is_vpn), ("nonvpn", ~is_vpn)]:
        if mask.sum() == 0:
            continue
        info  = _s2[key]
        X_sc  = info["scaler"].transform(X_all[mask])
        X_t   = torch.tensor(X_sc).unsqueeze(1)
        with torch.no_grad():
            preds = info["model"](X_t).argmax(1).numpy()
        labels = info["le"].inverse_transform(preds)
        if key == "vpn":
            labels = np.array([f"VPN-{l}" for l in labels])
        final_preds[mask] = labels
    return test["class1"].values, final_preds

data_b = get_data_b()
s1_model, s1_le, s2 = get_models()
y_true, y_pred = run_pipeline(s1_model, s1_le, s2)
overall_acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.title("🔍 Encrypted Network Traffic Classifier")
st.markdown(
    "A machine learning system that identifies **what type of traffic is on your network** "
    "— even when it's encrypted or tunneled through a VPN — using only behavioral flow statistics. "
    "No packet inspection. No privacy violations."
)
st.divider()

# ── Key metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall Accuracy", f"{overall_acc:.1%}", "14-class classification")
c2.metric("VPN Detection", "94%", "Stage 1 Random Forest")
c3.metric("VPN Traffic Types", "78%", "Stage 2 CNN, 7 classes")
c4.metric("Non-VPN Traffic Types", "81%", "Stage 2 CNN, 7 classes")
c5.metric("Dataset", "88k flows", "ISCXVPN2016")
st.divider()

# ── How it works ──────────────────────────────────────────────────────────────
st.subheader("How It Works")
col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("""
    Modern network traffic is encrypted end-to-end. Traditional deep packet inspection
    (DPI) tools are blind to encrypted payloads and raise serious privacy concerns.

    This system takes a different approach — it classifies traffic purely from
    **flow-level behavioral statistics**: timing patterns, packet rates, byte counts.
    A VoIP call looks different from a file transfer even when both are encrypted,
    because the *rhythm* of the traffic differs.

    **The pipeline runs in two stages:**

    1. **Stage 1 — VPN Detection** · Random Forest trained on Scenario A1.
       Detects whether a flow is VPN-tunneled with 94% accuracy.

    2. **Stage 2 — Traffic Type** · Two specialized 1D CNNs (one for VPN flows,
       one for plain flows), each classifying 7 traffic types.
       Splitting by VPN status lets each model specialize, dramatically improving
       VPN-encrypted traffic recognition from ~5% → ~76% recall.
    """)
with col_b:
    # Pipeline diagram as a simple chart
    stages = ["Raw Flow\n(23 features)", "Stage 1\nVPN Detection\n(RF · 94%)",
              "Stage 2a\nVPN CNN\n(78%)", "Stage 2b\nNon-VPN CNN\n(81%)", "Final\nPrediction"]
    fig, ax = plt.subplots(figsize=(3, 5))
    for i, s in enumerate(stages):
        color = "#e74c3c" if "VPN" in s and "Non" not in s else "#3498db" if "Non" in s else "#2c3e50"
        ax.text(0.5, 1 - i * 0.22, s, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8, edgecolor="white"),
                color="white", transform=ax.transAxes)
        if i < len(stages) - 1:
            ax.annotate("", xy=(0.5, 0.78 - i * 0.22), xytext=(0.5, 0.82 - i * 0.22),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="gray"))
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Model comparison ──────────────────────────────────────────────────────────
st.subheader("Why Hierarchical? — Model Comparison")
col1, col2 = st.columns([1, 2])
with col1:
    comparison = pd.DataFrame({
        "Model": ["Flat RF", "Flat CNN", "Hierarchical CNN"],
        "Features": [13, 13, 23],
        "Overall Acc": ["54.1%", "60.0%", f"{overall_acc:.1%}"],
        "VPN Recall": ["~11%", "~5%", "~76%"],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption(
        "Overall accuracy is similar across models, but the hierarchical approach "
        "dramatically improves recall on VPN-encrypted traffic — the hardest and most "
        "practically relevant problem."
    )

with col2:
    # VPN class recall comparison bar chart
    vpn_classes = [c for c in sorted(set(y_true)) if c.startswith("VPN-")]
    flat_rf_recall  = [0.12, 0.08, 0.10, 0.23, 0.15, 0.11, 0.09]
    flat_cnn_recall = [0.10, 0.04, 0.14, 0.18, 0.00, 0.05, 0.04]
    hier_recall     = [report.get(c, {}).get("recall", 0) for c in vpn_classes]

    x = np.arange(len(vpn_classes))
    w = 0.26
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(x - w, flat_rf_recall,  w, label="Flat RF",    color="#95a5a6")
    ax2.bar(x,     flat_cnn_recall, w, label="Flat CNN",   color="#3498db")
    ax2.bar(x + w, hier_recall,     w, label="Hierarchical", color="#e74c3c")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("VPN-", "") for c in vpn_classes], rotation=30, ha="right")
    ax2.set_ylabel("Recall")
    ax2.set_ylim(0, 1)
    ax2.set_title("VPN Traffic Type Recall — All Models")
    ax2.legend()
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ── Per-class performance ─────────────────────────────────────────────────────
st.subheader("Per-Class Performance — Hierarchical Pipeline")
classes = sorted(set(y_true))
perf_df = pd.DataFrame({
    "Traffic Type":  classes,
    "Precision":     [report.get(c, {}).get("precision", 0) for c in classes],
    "Recall":        [report.get(c, {}).get("recall", 0) for c in classes],
    "F1 Score":      [report.get(c, {}).get("f1-score", 0) for c in classes],
    "Test Samples":  [int(report.get(c, {}).get("support", 0)) for c in classes],
    "VPN Encrypted": ["Yes" if c.startswith("VPN-") else "No" for c in classes],
}).set_index("Traffic Type").round(2)

st.dataframe(
    perf_df.style.background_gradient(subset=["F1 Score"], cmap="RdYlGn"),
    use_container_width=True,
)
st.divider()

# ── Tech stack ────────────────────────────────────────────────────────────────
st.subheader("Technical Details")
t1, t2, t3 = st.columns(3)
t1.markdown("""
**Models**
- Random Forest (scikit-learn)
- 1D Convolutional Neural Network (PyTorch)
- Hierarchical two-stage pipeline
- Early stopping with validation loss monitoring
""")
t2.markdown("""
**Features**
- 23 flow-level behavioral statistics
- Inter-arrival times, packet/byte rates
- Active/idle period statistics
- No payload content — privacy preserving
""")
t3.markdown("""
**Dataset & Scale**
- ISCXVPN2016 (University of New Brunswick)
- 88,382 labeled flows, 14 traffic classes
- 7 plain + 7 VPN-encrypted traffic types
- 80/20 stratified train/test split
""")

st.divider()
st.caption("Isaac Mei · ECE 57000 · Purdue University · 2025 | Navigate to **Live Traffic Monitor** in the sidebar to see the classifier in action.")
