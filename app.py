from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv

import streamlit as st

from src.predictor import load_model, predict_image

st.set_page_config(
    page_title="Rice Leaf Blast Detector",
    page_icon="🌾",
    layout="centered",
)

MODEL_PATH_DEFAULT = Path("models") / "rice_leaf_blast_cnn.keras"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
PRED_LOG = REPORTS_DIR / "predictions.csv"


def write_log(row: dict):
    write_header = not PRED_LOG.exists()
    with open(PRED_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def guidance(mode: str, predicted: str, prob_blast: float):
    # Keep it safe: no chemical prescriptions, no medical-like certainty.
    if mode == "Farmer":
        if predicted == "blast":
            return [
                "⚠️ Your leaf image looks like **Leaf Blast**.",
                "✅ **Next steps:** isolate affected plants if possible and contact an **Agriculture Officer** for advice.",
                "📌 Keep monitoring nearby plants for similar spots/lesions.",
            ]
        else:
            return [
                "✅ Your leaf image looks **Healthy**.",
                "📌 Keep monitoring weekly, especially after heavy rain or high humidity.",
            ]

    if mode == "Agriculture Officer":
        if predicted == "blast":
            return [
                f"Result: **Leaf Blast** (confidence indicator: {prob_blast:.2f})",
                "Suggested follow-up: confirm in field, check recent weather/humidity, and inspect surrounding plots.",
                "Note: model is image-based; field conditions and cultivar variation can affect performance.",
            ]
        else:
            return [
                "Result: **Healthy**",
                "Suggested follow-up: routine monitoring; consider capturing multiple leaves if symptoms are suspected.",
            ]

    if mode == "Student":
        return [
            "This is a **binary classifier**: `blast` vs `healthy`.",
            "Model output is interpreted as probabilities shown below.",
            "We used a clean deduplicated split and evaluated using confusion matrix + report.",
        ]

    # Demo mode
    if predicted == "blast":
        return [
            "🌾 Demo Result: **Leaf Blast detected**",
            "This demo shows end-to-end ML: clean data split, trained model, and deployment UI.",
        ]
    return [
        "🌾 Demo Result: **Healthy leaf detected**",
        "This demo shows end-to-end ML: clean data split, trained model, and deployment UI.",
    ]


@st.cache_resource
def get_model(model_path: str):
    return load_model(model_path)


st.title("🌾 Rice Leaf Blast Detector")
st.caption("Phase 2: Multi-user demo (Farmer / Officer / Student / Demo) — using your trained model.")

mode = st.selectbox("Select user mode", ["Farmer", "Agriculture Officer", "Student", "Demo"], index=0)

with st.expander("📌 Model Card (read me)", expanded=(mode in ["Student", "Demo"])):
    st.markdown("""
**Task:** Binary classification — `blast` vs `healthy` (Rice Leaf Blast)

**Dataset:** "Rice Leaf Bacterial and Fungal Disease Dataset" (Original images only)

**Evaluation:** Clean split after removing exact duplicate images (hash-based deduplication)

**Clean Test Result (deduplicated):**
- Accuracy: **0.94**
- Confusion matrix (rows=true, cols=pred):  
  `[[38, 4], [0, 24]]`

**Error pattern (important):**
- Some blast images are predicted as healthy (false negatives).  
- Fewer false alarms on healthy leaves.

**Limitations:**
- Image quality, lighting, and camera angle can reduce performance.
- Not a medical/agronomic diagnosis — use as decision support and consult an officer if unsure.
""")


st.divider()

with st.expander("Settings", expanded=False):
    model_path = st.text_input("Model path", value=str(MODEL_PATH_DEFAULT))
    threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    st.caption("Threshold rule: predict **healthy** if prob_healthy ≥ threshold, else **blast**. Lower threshold = more likely to predict healthy.")


uploaded = st.file_uploader("Upload a rice leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    # Save upload to a temp file (streamlit provides bytes)
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
        st.json({
            "prob_blast": round(pred.prob_blast, 4),
            "prob_healthy": round(pred.prob_healthy, 4),
            "threshold": pred.threshold
        })

        st.write("**Guidance**")
        for line in guidance(mode, pred.predicted, pred.prob_blast):
            st.write("-", line)

        # Log for Officer + Demo
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
else:
    st.info("Upload an image to begin.")
