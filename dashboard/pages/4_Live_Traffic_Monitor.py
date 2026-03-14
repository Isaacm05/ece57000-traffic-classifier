import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import time
from data_utils import load_data, clean_data, FEATURE_COLS, LABEL_COL
from models.stage2_traffic_classification.train_stage2 import TrafficCNN

st.set_page_config(page_title="Live Traffic Monitor", layout="wide")
st.title("🖥️ Live Traffic Monitor")
st.markdown("Simulates real-time network flow classification. Flows are fed through the hierarchical pipeline one batch at a time.")

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    data = load_data("data/scenario_b")
    data = clean_data(data)
    data["class1"] = data["class1"].str.strip()
    return data

@st.cache_data
def build_session_stream(max_flows, session_min, session_max, mode="Non-VPN → VPN", seed=42):
    """
    Build a realistic traffic stream in two phases:
      Phase 1 — plain traffic sessions (BROWSING, VOIP, CHAT, etc.)
      Phase 2 — VPN traffic sessions (VPN-VOIP, VPN-P2P, etc.)
    This simulates a user running normal traffic then switching on a VPN.
    """
    data = get_data()
    rng = np.random.default_rng(seed)

    vpn_classes    = sorted([c for c in data["class1"].unique() if c.startswith("VPN-")])
    nonvpn_classes = sorted([c for c in data["class1"].unique() if not c.startswith("VPN-")])

    if mode == "Non-VPN → VPN":
        phase_order = [nonvpn_classes, vpn_classes]
    elif mode == "VPN → Non-VPN":
        phase_order = [vpn_classes, nonvpn_classes]
    else:  # Mixed
        phase_order = [nonvpn_classes + vpn_classes]

    half = max_flows // len(phase_order)
    stream_parts = []

    for phase_classes in phase_order:
        total = 0
        while total < half:
            traffic_class = rng.choice(phase_classes)
            session_len   = int(rng.integers(session_min, session_max + 1))
            subset = data[data["class1"] == traffic_class]
            if len(subset) == 0:
                continue
            sample = subset.sample(n=min(session_len, len(subset)), replace=True,
                                   random_state=int(rng.integers(0, 9999)))
            stream_parts.append(sample)
            total += len(sample)

    stream = pd.concat(stream_parts, ignore_index=True).head(max_flows)
    # Add session ID column for display
    session_ids = []
    sid = 0
    prev_class = None
    for c in stream["class1"]:
        if c != prev_class:
            sid += 1
            prev_class = c
        session_ids.append(sid)
    stream["_session_id"] = session_ids
    return stream

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

def classify_batch(X_batch, stage1_model, stage1_le, stage2):
    """Run full hierarchical pipeline on a batch of flows."""
    X = X_batch.astype(np.float32)
    vpn_labels = stage1_le.inverse_transform(stage1_model.predict(X))
    vpn_probas = stage1_model.predict_proba(X).max(axis=1)
    is_vpn = vpn_labels == "VPN"

    final_labels = np.empty(len(X), dtype=object)
    final_confs  = np.zeros(len(X))

    for key, mask in [("vpn", is_vpn), ("nonvpn", ~is_vpn)]:
        if mask.sum() == 0:
            continue
        info  = stage2[key]
        X_sc  = info["scaler"].transform(X[mask])
        X_t   = torch.tensor(X_sc).unsqueeze(1)
        with torch.no_grad():
            logits = info["model"](X_t)
            probs  = torch.softmax(logits, dim=1).numpy()
        preds  = probs.argmax(axis=1)
        labels = info["le"].inverse_transform(preds)
        confs  = probs.max(axis=1)
        if key == "vpn":
            labels = np.array([f"VPN-{l}" for l in labels])
        final_labels[mask] = labels
        final_confs[mask]  = confs

    return vpn_labels, vpn_probas, final_labels, final_confs

TYPE_COLORS = {
    "BROWSING": "#3498db", "CHAT": "#2ecc71", "FT": "#e67e22",
    "MAIL": "#9b59b6", "P2P": "#e74c3c", "STREAMING": "#1abc9c", "VOIP": "#f39c12",
    "VPN-BROWSING": "#2980b9", "VPN-CHAT": "#27ae60", "VPN-FT": "#d35400",
    "VPN-MAIL": "#8e44ad", "VPN-P2P": "#c0392b", "VPN-STREAMING": "#16a085", "VPN-VOIP": "#f1c40f",
}

data = get_data()
stage1_model, stage1_le = get_stage1()
stage2 = get_stage2()

# ── Controls ──────────────────────────────────────────────────────────────────
st.subheader("Controls")
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

source = col1.selectbox("Data source", ["ISCXVPN2016 — Scenario B (session-based)", "Upload ARFF file"])
uploaded = None
if source == "Upload ARFF file":
    uploaded = col1.file_uploader("Upload ARFF", type=["arff"])

batch_size   = col2.slider("Flows per batch", min_value=1, max_value=20, value=5)
speed        = col3.slider("Delay between batches (s)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
max_flows    = col4.slider("Max flows to process", min_value=50, max_value=500, value=200, step=50)

col5, col6, col7 = st.columns([2, 3, 2])
session_min  = col5.slider("Min flows per session", min_value=10, max_value=80, value=40)
session_max  = col6.slider("Max flows per session", min_value=20, max_value=200, value=100)
stream_mode  = col7.selectbox("Simulation mode", ["Non-VPN → VPN", "VPN → Non-VPN", "Mixed"])

col_start, col_stop = st.columns([1, 5])
start = col_start.button("▶ Start", type="primary")
stop  = col_start.button("⏹ Stop")

if stop:
    st.session_state["running"] = False
if start:
    st.session_state["running"] = True
    st.session_state["log"] = pd.DataFrame()

# ── Layout placeholders ───────────────────────────────────────────────────────
st.divider()
stat_row = st.columns(4)
ph_total    = stat_row[0].empty()
ph_vpn      = stat_row[1].empty()
ph_acc      = stat_row[2].empty()
ph_toptype  = stat_row[3].empty()

chart_col1, chart_col2, chart_col3 = st.columns(3)
ph_pie    = chart_col1.empty()
ph_bar    = chart_col2.empty()
ph_conf   = chart_col3.empty()

st.subheader("Flow Feed")
ph_session = st.empty()
ph_table = st.empty()

# ── Simulation loop ───────────────────────────────────────────────────────────
if st.session_state.get("running"):
    if uploaded:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".arff", delete=False) as f:
            f.write(uploaded.read())
            tmp_path = f.name
        sim_data = clean_data(load_data(os.path.dirname(tmp_path)))
        sim_data["class1"] = sim_data["class1"].str.strip()
        sim_data["_session_id"] = 0
    else:
        sim_data = build_session_stream(max_flows, session_min, session_max, mode=stream_mode)

    log_rows = []

    for batch_start in range(0, len(sim_data), batch_size):
        if not st.session_state.get("running"):
            break

        batch = sim_data.iloc[batch_start:batch_start + batch_size]
        X_batch = batch[FEATURE_COLS].values

        vpn_labels, vpn_probas, final_labels, final_confs = classify_batch(
            X_batch, stage1_model, stage1_le, stage2
        )

        for i in range(len(batch)):
            true_label = batch.iloc[i]["class1"]
            session_id = batch.iloc[i].get("_session_id", "-")
            log_rows.append({
                "Session":    f"#{int(session_id)}",
                "Flow #":     batch_start + i + 1,
                "VPN?":       "🔴 VPN" if vpn_labels[i] == "VPN" else "🟢 Non-VPN",
                "Predicted":  final_labels[i],
                "True Label": true_label,
                "✓":          "✓" if final_labels[i] == true_label else "✗",
                "Confidence": f"{final_confs[i]:.0%}",
                "Bytes/s":    f"{float(batch.iloc[i]['flowBytesPerSecond']):,.0f}",
                "Pkts/s":     f"{float(batch.iloc[i]['flowPktsPerSecond']):.1f}",
                "_conf_val":  final_confs[i],
                "_predicted": final_labels[i],
                "_vpn":       vpn_labels[i] == "VPN",
                "_correct":   final_labels[i] == true_label,
            })

        log_df = pd.DataFrame(log_rows)

        # ── Stats ─────────────────────────────────────────────────────────────
        total    = len(log_df)
        vpn_pct  = log_df["_vpn"].mean()
        accuracy = log_df["_correct"].mean()
        top_type = log_df["_predicted"].value_counts().idxmax()

        ph_total.metric("Flows Processed", total)
        ph_vpn.metric("VPN Traffic", f"{vpn_pct:.1%}")
        ph_acc.metric("Classifier Accuracy", f"{accuracy:.1%}")
        ph_toptype.metric("Top Traffic Type", top_type.replace("VPN-", ""))

        # ── Pie chart: VPN vs Non-VPN ─────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        vpn_counts = log_df["_vpn"].value_counts()
        ax1.pie(
            [vpn_counts.get(True, 0), vpn_counts.get(False, 0)],
            labels=["VPN", "Non-VPN"],
            colors=["#e74c3c", "#3498db"],
            autopct="%1.0f%%", startangle=90,
        )
        ax1.set_title("VPN vs Non-VPN")
        plt.tight_layout()
        ph_pie.pyplot(fig1)
        plt.close(fig1)

        # ── Bar chart: traffic type distribution ──────────────────────────────
        type_counts = log_df["_predicted"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        colors = [TYPE_COLORS.get(t, "#95a5a6") for t in type_counts.index]
        ax2.barh(type_counts.index, type_counts.values, color=colors)
        ax2.set_xlabel("Count")
        ax2.set_title("Traffic Type Distribution")
        plt.tight_layout()
        ph_bar.pyplot(fig2)
        plt.close(fig2)

        # ── Confidence over time ──────────────────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.plot(log_df["Flow #"], log_df["_conf_val"], color="#9b59b6", linewidth=0.8, alpha=0.7)
        ax3.axhline(log_df["_conf_val"].mean(), color="orange", linestyle="--", label=f"avg {log_df['_conf_val'].mean():.0%}")
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("Flow #"); ax3.set_ylabel("Confidence")
        ax3.set_title("Classification Confidence")
        ax3.legend()
        plt.tight_layout()
        ph_conf.pyplot(fig3)
        plt.close(fig3)

        # ── Current session banner ────────────────────────────────────────────
        current = log_rows[-1]
        is_vpn_str = "🔴 VPN" if current["_vpn"] else "🟢 Non-VPN"
        ph_session.info(f"**Active Session #{current['Session']}** — True type: `{current['True Label']}` | {is_vpn_str}")

        # ── Flow table (most recent 20) ───────────────────────────────────────
        display_cols = ["Session", "Flow #", "VPN?", "Predicted", "True Label", "✓", "Confidence", "Bytes/s", "Pkts/s"]
        ph_table.dataframe(
            log_df[display_cols].tail(20).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        time.sleep(speed)

    st.session_state["running"] = False
    st.success(f"✓ Simulation complete — {len(log_rows)} flows processed.")
