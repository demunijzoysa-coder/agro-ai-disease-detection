from __future__ import annotations

import sys
from pathlib import Path
import csv
from datetime import datetime

import pandas as pd
import streamlit as st

# -----------------------------
# Path setup (works on Streamlit Cloud)
# -----------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # repo root
sys.path.insert(0, str(ROOT / "src"))  # allow: import predictor

from predictor import load_model, predict_image  # noqa: E402


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Rice Leaf Blast Detector",
    page_icon="🌾",
    layout="centered",
)

MODEL_PATH_DEFAULT = Path("models") / "rice_leaf_blast_cnn.keras"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
PRED_LOG = REPORTS_DIR / "predictions.csv"

SAT_CSV = Path("data") / "satellite" / "risk_features.csv"


def write_log(row: dict) -> None:
    """Append predictions to a local CSV (runtime log)."""
    write_header = not PRED_LOG.exists()
    with open(PRED_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def guidance(mode: str, predicted: str, prob_blast: float) -> list[str]:
    """Mode-specific safe guidance (decision support, not diagnosis)."""
    if mode == "Farmer":
        if predicted == "blast":
            return [
                "⚠️ This leaf image looks like **Leaf Blast**.",
                "✅ **Next steps:** isolate affected plants if possible and contact an **Agriculture Officer** for advice.",
                "📌 Monitor nearby plants for similar symptoms.",
            ]
        return [
            "✅ This leaf image looks **Healthy**.",
            "📌 Keep monitoring weekly, especially after heavy rain or high humidity.",
        ]

    if mode == "Agriculture Officer":
        if predicted == "blast":
            return [
                f"Result: **Leaf Blast** (confidence indicator: {prob_blast:.2f})",
                "Suggested follow-up: confirm in field, review recent humidity/rain patterns, inspect surrounding plots.",
                "Note: model is image-based; cultivar variation and lighting can affect performance.",
            ]
        return [
            "Result: **Healthy**",
            "Suggested follow-up: routine monitoring; if unsure, capture multiple leaves from different areas.",
        ]

    if mode == "Student":
        return [
            "Binary classifier: `blast` vs `healthy`.",
            "Sigmoid output represents probability for class **healthy**; P(blast) = 1 − P(healthy).",
            "Threshold rule: predict healthy if P(healthy) ≥ threshold; else blast.",
        ]

    # Demo
    if predicted == "blast":
        return [
            "🌾 Demo Result: **Leaf Blast detected**",
            "This demo shows end-to-end ML: clean split, dedup checks, deployment UI.",
        ]
    return [
        "🌾 Demo Result: **Healthy leaf detected**",
        "This demo shows end-to-end ML: clean split, dedup checks, deployment UI.",
    ]


@st.cache_resource
def get_model(model_path: str):
    """Cache model load for speed."""
    return load_model(model_path)


# -----------------------------
# Header
# -----------------------------
st.title("🌾 Rice Leaf Blast Detector")
st.caption("Multi-modal demo: Satellite screening (NDVI risk) → Leaf photo confirmation (CNN).")

# User mode + settings
mode = st.selectbox("Select user mode", ["Farmer", "Agriculture Officer", "Student", "Demo"], index=0)

with st.expander("📌 Model Card (read me)", expanded=(mode in ["Student", "Demo"])):
    st.markdown(
        """
**Task:** Binary classification — `blast` vs `healthy` (Rice Leaf Blast)

**Dataset:** Rice leaf disease dataset (original images)

**Data Integrity:** Exact duplicate images removed using hash-based deduplication before clean split

**Clean Test Result (deduplicated):**
- Accuracy: **0.94**
- Confusion matrix (rows=true, cols=pred): `[[38, 4], [0, 24]]`

**Limitations:**
- Sensitive to lighting, blur, angles
- Decision support only (not diagnosis)
"""
    )

with st.expander("Settings", expanded=False):
    model_path = st.text_input("Model path", value=str(MODEL_PATH_DEFAULT))
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    st.caption("Predict **healthy** if P(healthy) ≥ threshold; else **blast**.")

st.divider()

tabs = st.tabs(["🛰️ Satellite Risk", "🍃 Leaf Check"])


# -----------------------------
# TAB 1: Satellite Risk
# -----------------------------
with tabs[0]:
    st.subheader("🛰️ Satellite Risk (Real Sentinel-2 NDVI MVP)")
    st.caption("Satellite indicates vegetation stress patterns, not direct disease detection.")

    if not SAT_CSV.exists():
        st.warning(
            "Satellite CSV not found: `data/satellite/risk_features.csv`\n\n"
            "Generate it locally using:\n"
            "`python src/satellite_ndvi_mvp.py`"
        )
    else:
        df = pd.read_csv(SAT_CSV)

        if df.empty:
            st.warning("Satellite CSV exists but contains no rows.")
        else:
            # Parse + sort by date
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")

            latest = df.iloc[-1]
            latest_ndvi = float(latest.get("ndvi", 0.0))
            risk_score = float(latest.get("risk_score", 0.0))
            risk_band = str(latest.get("risk_band", "UNKNOWN")).upper()

            c1, c2, c3 = st.columns(3)
            c1.metric("Latest NDVI", f"{latest_ndvi:.3f}")
            c2.metric("Risk Score (0–100)", f"{risk_score:.1f}")
            c3.metric("Risk Band", risk_band)

            st.write("### Trends")
            st.line_chart(df.set_index("date")[["ndvi"]])
            st.line_chart(df.set_index("date")[["risk_score"]])

            st.write("### Recommendation")
            if risk_band == "HIGH":
                st.error("High risk detected. Capture 3–5 leaf photos across the field and confirm in the Leaf Check tab.")
            elif risk_band == "MEDIUM":
                st.warning("Medium risk. Monitor conditions; capture leaf photos if symptoms appear.")
            elif risk_band == "LOW":
                st.success("Low risk. Continue routine monitoring.")
            else:
                st.info("Risk band is unknown. Verify satellite CSV columns/values.")

            with st.expander("View raw satellite features (CSV)", expanded=False):
                st.dataframe(df, use_container_width=True)


# -----------------------------
# TAB 2: Leaf Check
# -----------------------------
with tabs[1]:
    st.subheader("🍃 Leaf Check (Photo Confirmation)")
    uploaded = st.file_uploader("Upload a rice leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to begin.")
    else:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

        tmp_dir = Path(".tmp")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getbuffer())

        if st.button("Analyze", type="primary"):
            model = get_model(model_path)
            pred = predict_image(model, tmp_path, threshold=threshold)

            st.subheader("Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", pred.predicted.upper())
            with col2:
                st.metric("P(blast)", f"{pred.prob_blast:.3f}")

            st.progress(min(max(pred.prob_blast, 0.0), 1.0))

            st.write("**Probabilities**")
            st.json(
                {
                    "prob_blast": round(pred.prob_blast, 4),
                    "prob_healthy": round(pred.prob_healthy, 4),
                    "threshold": pred.threshold,
                }
            )

            st.write("**Guidance**")
            for line in guidance(mode, pred.predicted, pred.prob_blast):
                st.write("-", line)

            # Log only for Officer/Demo (runtime file; usually ignored by git)
            if mode in {"Agriculture Officer", "Demo"}:
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "mode": mode,
                    "filename": uploaded.name,
                    "predicted": pred.predicted,
                    "prob_blast": f"{pred.prob_blast:.6f}",
                    "prob_healthy": f"{pred.prob_healthy:.6f}",
                    "threshold": f"{pred.threshold:.2f}",
                }
                write_log(row)
                st.success("Saved prediction log to reports/predictions.csv")
                if PRED_LOG.exists():
                    st.download_button(
                        "Download predictions.csv",
                        data=PRED_LOG.read_bytes(),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
