#  Rice Leaf Blast Detection System (AI-Based)

An end-to-end AI system for **early detection of Rice Leaf Blast disease** using leaf images.  
Designed as a **decision-support tool** for farmers, agriculture officers, students, and demonstrations.

This project emphasizes **clean data practices, honest evaluation, and real usability**, not just model accuracy.

---

##  Problem Statement

Rice Leaf Blast is one of the most destructive fungal diseases affecting rice crops, especially in South and Southeast Asia.  
Late detection can significantly reduce yield and spread disease across fields.

Manual inspection is:
- Time-consuming
- Subjective
- Not always accessible to smallholder farmers

---

##  Solution Overview

This project provides an **AI-powered image classification system** that:
- Analyzes rice leaf images
- Detects **Leaf Blast vs Healthy**
- Outputs confidence scores
- Offers safe, non-prescriptive guidance

The system is implemented as:
- A trained CNN model
- A command-line inference tool
- A multi-user web application (Streamlit)

---

##  Supported User Modes

The web application adapts its output based on the selected user:

- **Farmer**
  - Simple language
  - Clear next steps
- **Agriculture Officer**
  - Confidence indicators
  - Logging & CSV export
- **Student**
  - ML interpretation details
  - Threshold logic explanation
- **Demo**
  - Clean presentation
  - Model facts & performance summary

---

##  Machine Learning Approach

- **Task:** Binary image classification (`blast` vs `healthy`)
- **Model:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input size:** 224 × 224 RGB images
- **Output:** Sigmoid probability (interpreted correctly)

### Dataset
- Source: *Rice Leaf Bacterial and Fungal Disease Dataset*
- Used only **original images**
- **Exact duplicate images removed using hash-based deduplication**
- Clean train / validation / test split

---

##  Evaluation Results (Clean Split)

After deduplication:

- **Test Accuracy:** **94%**
- **Confusion Matrix** (rows = true, columns = predicted):

[[38, 4],
[ 0, 24]]


### Interpretation
- Strong detection of Leaf Blast
- Conservative behavior (few false alarms)
- Honest evaluation without data leakage

---

##  Limitations & Ethics

- Performance depends on image quality and lighting
- Not a medical or agronomic diagnosis
- Intended for **decision support only**
- Users should consult agriculture officers for treatment decisions

---

##  How to Run

### 1️ Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

### 2 Run CLI Inference
python src/infer.py "path\to\leaf_image.jpg"

Adjust decision threshold
python src/infer.py "path\to\leaf_image.jpg" --threshold 0.7

### 3 Run Web Application (Streamlit)

## Project Structure

agro-ai-disease-detection/
│
├── app.py                 # Streamlit web app
├── src/
│   ├── predictor.py       # Shared inference logic
│   └── infer.py           # CLI inference tool
├── models/
│   └── rice_leaf_blast_cnn.keras
├── notebooks/             # Training & analysis notebooks
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── predictions.csv
├── requirements.txt
└── README.md
