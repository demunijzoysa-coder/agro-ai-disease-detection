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
## 6.1 ASCII Architecture Diagram

![alt text](<ASCII Architecture Diagram.png>)

# Risk Score Formula Baseline (rule-based MVP)

## 6.2 Baseline Risk Score (Rule-Based MVP)

Before training any satellite ML model, we start with a transparent baseline score.
The score is designed to be **interpretable** and safe (decision support, not diagnosis).

### Inputs (per field/region tile)
- **ΔNDVI**: recent NDVI change (drop indicates stress)
- **Rain_7d**: total rainfall in last 7 days
- **Temp_7d**: mean temperature in last 7 days
- **Humidity proxy** (optional): if available; otherwise approximate using rainfall + temperature patterns
- **Cloud_coverage**: used to ignore unreliable satellite observations

### Normalization
Convert each raw feature into a 0..1 risk component using simple clipping:

- NDVI stress component:
  - `ndvi_drop = clip((NDVI_prev - NDVI_now) / 0.15, 0, 1)`
  - (0.15 is a tunable “large drop” scale)

- Rain component:
  - `rain = clip(Rain_7d / 120, 0, 1)`  
  - (120mm is a tunable “very wet week” scale)

- Temperature component (example sweet spot for fungal growth):
  - `temp = 1 - clip(abs(Temp_7d - 26) / 8, 0, 1)`  
  - (center 26°C, tolerance 8°C; tune based on agronomy references)

### Baseline Score (0..100)
A simple weighted sum:

- `Risk = 100 * (0.45*ndvi_drop + 0.35*rain + 0.20*temp)`

### Risk Bands (UI)
- **0–39:** Low
- **40–69:** Medium
- **70–100:** High → recommend leaf-photo sampling for confirmation

### Notes
- This baseline does NOT claim “disease detection from space.”
- It is a **stress + context** indicator to prioritize inspections and guide sampling.
- Later, once ground truth labels exist, replace weights/rules with an ML model and evaluate properly.

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
## 7.1 Minimal Data Schema (MVP)

The satellite phase MVP can store features and scores in a simple tabular format (CSV/Parquet).
This keeps the pipeline transparent and easy to debug.

### Table: `risk_features` (one row per region/tile per date)

| Column              | Type    | Example                  | Description |
|---------------------|---------|--------------------------|-------------|
| tile_id             | string  | `tile_0012`              | Region/tile identifier (or field_id if polygons exist) |
| date                | date    | `2025-12-23`             | Observation date |
| ndvi_now            | float   | `0.62`                   | NDVI at date |
| ndvi_prev           | float   | `0.71`                   | Previous valid NDVI observation |
| ndvi_drop           | float   | `0.60`                   | Normalized (0..1) stress component |
| rain_7d_mm          | float   | `85.0`                   | Total rainfall last 7 days (mm) |
| rain_component      | float   | `0.71`                   | Normalized (0..1) |
| temp_7d_c           | float   | `27.4`                   | Mean temp last 7 days (°C) |
| temp_component      | float   | `0.93`                   | Normalized (0..1) |
| cloud_coverage      | float   | `0.18`                   | Fraction (0..1); used to filter observations |
| risk_score          | float   | `73.2`                   | Final risk score (0..100) |
| risk_band           | string  | `HIGH`                   | LOW / MEDIUM / HIGH |
| note                | string  | `ndvi drop + wet week`   | Optional human-readable reason |

### Optional Table: `ground_truth_labels` (future)
| Column   | Type   | Example | Description |
|----------|--------|---------|-------------|
| tile_id  | string | tile_0012 | Match to risk_features |
| date     | date   | 2025-12-23 | Label date |
| blast_observed | int | 1 | Officer-confirmed label (0/1) |
| source   | string | officer_visit | Where label came from |


---
## 7.2 Mock UI (Streamlit Satellite Risk Tab)

The satellite extension should be presented as a **risk indicator** (not diagnosis) and connected to the leaf-photo confirmation workflow.

### UI Layout (MVP)

**Tab: “Satellite Risk”**
1) **Region selector**
   - dropdown: `tile_id` (or field/cooperative name)
2) **Date selector**
   - select observation date (or “latest”)
3) **Risk indicator card**
   - Risk Band: LOW / MEDIUM / HIGH
   - Risk Score: 0..100
   - Short reason text: “NDVI dropped + wet week”
4) **Trend mini-chart (optional)**
   - NDVI over last N observations
   - Risk score trend over time
5) **Action prompt**
   - If HIGH:
     - “High risk detected. Capture 3–5 leaf photos from different areas for confirmation.”
     - Button: “Go to Leaf Check” (switch to existing leaf classifier tab)
   - If MEDIUM:
     - “Monitor. Consider sampling if symptoms appear.”
   - If LOW:
     - “Low risk. Continue routine monitoring.”

**Tab: “Leaf Check” (Existing)**
- Upload photo → predict blast/healthy
- Show confidence + guidance
- Officer/Demo modes can log outputs

### UX Principles
- Keep language simple for farmers
- Officers can see more details (components + thresholds)
- Students can see full feature breakdown and formula
- Always include limitation note: “Satellite indicates stress patterns, not direct disease detection.”


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
