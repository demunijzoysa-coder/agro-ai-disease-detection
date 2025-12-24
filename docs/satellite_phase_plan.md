# Satellite Phase Plan (Future Work / System Design)

## 1. Objective
Extend the current Rice Leaf Blast leaf-image classifier into a larger decision-support system using:
- Satellite imagery (field-level monitoring)
- Weather data (risk context)
- Leaf-photo model (confirmation step)

This is NOT intended to replace experts. It provides early warnings and triage support.

---

## 2. Why Satellite Data?
Leaf images are great for confirming disease on a specific plant, but they do not answer:
- Which fields are at highest risk?
- Where should officers inspect first?
- Are plants under stress (water, heat, disease) at a larger scale?

Satellite data provides a wide-area, repeatable view of crop condition.

---

## 3. Intended Users and Decisions
### Farmers
- "My crop might be stressed / at risk" → capture leaf photo for confirmation
- Basic monitoring and guidance, low digital literacy UI

### Agriculture Officers
- Prioritize inspections across fields/cooperatives
- Track risk zones over time
- Record outcomes for feedback loops

### Students / Demo Audience
- Demonstrates multi-modal AI + real-world pipeline thinking

---

## 4. Data Sources (Planned)
### Satellite imagery
- Sentinel-2 (10m–20m resolution; frequent revisits)
- Landsat (30m resolution; longer history)

### Weather data
- Rainfall
- Temperature
- Relative humidity (if available)
- Wind (optional)
- Derived features: rolling rainfall, humidity index, etc.

### Field boundaries (optional but ideal)
- GIS polygons for paddy fields or cooperatives
- If not available, use approximate grid-based regions

### Ground truth labels (needed for real satellite ML)
- Officer-confirmed disease occurrence by date & location
- Even weak labels are useful if properly handled

---

## 5. Core Features (What we compute)
### Vegetation indices (from multispectral imagery)
- NDVI: general vegetation vigor
- EVI: alternative vigor metric (better in dense vegetation)
- NDWI: water stress / moisture proxy (optional)

### Temporal features
- Change over time is often more informative than raw values:
  - NDVI drop week-over-week
  - Stress persistence over multiple observations

### Weather-driven risk features
- Blast risk often increases with specific humidity/rain patterns.
- Use rolling windows:
  - rainfall_last_3_days
  - rainfall_last_7_days
  - mean_temp_last_7_days
  - humidity_proxy_last_7_days (if available)

---

## 6. System Architecture (High Level)
### Two-stage decision support
1) Field-level risk screening (satellite + weather)
2) Leaf-level confirmation (phone image classifier)

Flow:
- Satellite/Weather → "Risk score" for a region/field
- If risk high → prompt for leaf photo sampling
- Leaf model predicts blast/healthy → supports next steps

---

## 7. Proposed Pipeline (MVP)
### Step A — Data collection (MVP)
- For a selected region (pilot area), gather:
  - Sentinel-2 images over time
  - Weather history (daily)
- Define regions:
  - Either known field polygons
  - Or a grid (e.g., 1km tiles)

### Step B — Feature extraction
- Compute NDVI (and optionally EVI/NDWI)
- Aggregate per region per date:
  - mean NDVI
  - NDVI change since last observation
- Join with weather features by date

### Step C — Risk scoring (Phase 1 satellite MVP)
Start with a simple model:
- Rule-based risk score (baseline)
- Then a lightweight ML model if labels exist:
  - logistic regression / XGBoost
  - temporal features

### Step D — Product integration
In the Streamlit app:
- Add a “Satellite Risk” tab:
  - Show a basic risk indicator for the selected region/date
  - If “High risk”, recommend capturing leaf images for confirmation

---

## 8. Evaluation Strategy
### Leaf model (already done)
- Clean split, dedup, confusion matrix, honest accuracy

### Satellite risk model (future)
Evaluation requires ground truth. Proposed approach:
- Officer validation:
  - region/date → disease observed yes/no
- Evaluate:
  - precision/recall for "high risk" alerts
  - calibration: does “high risk” correspond to higher disease rate?
- For early MVP without labels:
  - validate stress detection consistency (NDVI drops align with known events)
  - treat as a monitoring tool, not disease predictor

---

## 9. Risks and Limitations (Be honest)
- Satellite cannot directly “see” disease; it detects stress patterns
- Cloud cover reduces usable observations (especially in monsoon seasons)
- Field boundaries may be unavailable or inaccurate
- Requires ground-truth labeling effort to move from “stress monitoring” to “disease risk prediction”
- Must avoid overclaiming; keep messaging as decision support

---

## 10. Roadmap
### Satellite Phase 1 (Design + MVP monitoring)
- NDVI/EVI extraction for region tiles
- Weather feature integration
- Rule-based risk score
- UI: risk indicator + prompt for leaf photo sampling

### Satellite Phase 2 (True risk prediction)
- Collect ground truth labels with officer support
- Train ML model on time-series features
- Add uncertainty estimates + alert thresholds

### Satellite Phase 3 (Deployment)
- Automate ingestion and periodic updates
- Maintain a history dashboard (weekly trends)
- Integration with notifications (optional)

---

## 11. How This Strengthens the Portfolio
This satellite extension shows:
- Multi-modal AI system thinking
- Data engineering + ML pipeline design
- Ethical and realistic scoping
- Product integration mindset
