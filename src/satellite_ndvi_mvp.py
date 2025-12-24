import ee
import csv
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
PROJECT_ID = "agro-ai-disease-detection"

# Pilot area: Galewela region (approx)
LON, LAT = 80.56, 7.76
BUFFER_M = 2000  # 2 km radius

START_DATE = "2024-10-01"
END_DATE   = "2025-03-01"

OUT_CSV = Path("data/satellite/risk_features.csv")
OUT_PLOT = Path("reports/ndvi_risk_timeseries.png")

# ===============================
# INIT EARTH ENGINE
# ===============================
ee.Initialize(project=PROJECT_ID)

point = ee.Geometry.Point([LON, LAT])
region = point.buffer(BUFFER_M).bounds()

# ===============================
# SENTINEL-2 + NDVI
# ===============================
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 30))
)

def add_ndvi(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.addBands(ndvi)

s2 = s2.map(add_ndvi)

def img_to_feature(img):
    stats = img.select("NDVI").reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10,
        maxPixels=1e9,
    )
    return ee.Feature(None, {
        "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
        "ndvi": stats.get("NDVI"),
        "cloud": img.get("CLOUDY_PIXEL_PERCENTAGE"),
    })

fc = (
    ee.FeatureCollection(s2.map(img_to_feature))
    .filter(ee.Filter.notNull(["ndvi"]))
    .sort("date")
)

rows = fc.getInfo()["features"]

# ===============================
# RISK SCORE (BASELINE)
# ===============================
records = []
prev_ndvi = None

for f in rows:
    p = f["properties"]
    ndvi = p["ndvi"]
    date = p["date"]

    ndvi_drop = 0.0
    if prev_ndvi is not None:
        ndvi_drop = max(0.0, min((prev_ndvi - ndvi) / 0.15, 1.0))

    # Simple MVP risk score (no weather yet)
    risk_score = round(100 * ndvi_drop, 2)

    band = "LOW"
    if risk_score >= 70:
        band = "HIGH"
    elif risk_score >= 40:
        band = "MEDIUM"

    records.append({
        "date": date,
        "ndvi": round(ndvi, 4),
        "ndvi_drop": round(ndvi_drop, 3),
        "risk_score": risk_score,
        "risk_band": band,
    })

    prev_ndvi = ndvi

# ===============================
# SAVE CSV
# ===============================
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"✅ CSV saved to {OUT_CSV}")

# ===============================
# PLOT
# ===============================
dates = [r["date"] for r in records]
ndvis = [r["ndvi"] for r in records]
risks = [r["risk_score"] for r in records]

plt.figure(figsize=(10, 5))
plt.plot(dates, ndvis, marker="o", label="NDVI")
plt.xticks(rotation=45)
plt.ylabel("NDVI")
plt.title("Sentinel-2 NDVI Time Series (Galewela Pilot)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_PLOT)

print(f"📈 Plot saved to {OUT_PLOT}")
