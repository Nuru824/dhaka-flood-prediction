from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dhaka Flood Risk Predictor", page_icon="🌊", layout="centered")
st.title("🌊 Dhaka Next-Day Flood Risk Predictor")
st.caption("Random Forest classifier predicting next-day flood risk from cumulative rainfall, soil moisture deficit, and seasonal meteorological features.")

BASE_DIR = Path(__file__).parent
artifact = joblib.load(BASE_DIR / "dhaka_flood_model.joblib")
model = artifact["rf_model"]
FEATURES = artifact["features"]
THRESHOLD = artifact["threshold"]

# Quick Scenario Presets
preset = st.radio(
    "Load a sample weather scenario:",
    ["Custom Input", "Heavy Monsoon Flood Risk", "Dry Season Low Risk"],
    horizontal=True,
)

if preset == "Heavy Monsoon Flood Risk":
    defaults = [45.0, 110.0, 185.0, 310.0, 560.0, 35.0, 28.0, True, 210, -0.5, -0.86, 92.0, 2.5, 27.5]
elif preset == "Dry Season Low Risk":
    defaults = [0.0, 0.0, 2.0, 5.0, 12.0, -40.0, -15.0, False, 30, 0.5, 0.86, 62.0, -1.0, 21.0]
else:
    defaults = [25.0, 65.0, 140.0, 240.0, 450.0, 15.0, 12.0, True, 200, -0.5, -0.86, 88.0, 1.5, 28.5]

col1, col2 = st.columns(2)
with col1:
    prectotcorr = st.number_input("Today's Rainfall (mm)", value=defaults[0])
    rain_3day = st.number_input("3-Day Cumulative Rain (mm)", value=defaults[1])
    rain_7day = st.number_input("7-Day Cumulative Rain (mm)", value=defaults[2])
    rain_14day = st.number_input("14-Day Cumulative Rain (mm)", value=defaults[3])
    rain_30day = st.number_input("30-Day Cumulative Rain (mm)", value=defaults[4])
    moisture_deficit = st.number_input("Moisture Deficit", value=defaults[5])
    rain_anomaly = st.number_input("Rain Anomaly", value=defaults[6])

with col2:
    is_monsoon = st.checkbox("Is Monsoon Season?", value=defaults[7])
    day_of_year = st.slider("Day of Year", 1, 365, defaults[8])
    month_sin = st.number_input("Month Sin", value=defaults[9])
    month_cos = st.number_input("Month Cos", value=defaults[10])
    rh_3day_avg = st.number_input("3-Day Avg Humidity (%)", value=defaults[11])
    rh_trend = st.number_input("Humidity Trend", value=defaults[12])
    temp_3day_avg = st.number_input("3-Day Avg Temp (°C)", value=defaults[13])

if st.button("Predict Tomorrow's Flood Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([[
        prectotcorr, rain_3day, rain_7day, rain_14day, rain_30day,
        moisture_deficit, rain_anomaly, int(is_monsoon), day_of_year,
        month_sin, month_cos, rh_3day_avg, rh_trend, temp_3day_avg
    ]], columns=FEATURES)

    prob = model.predict_proba(input_df)[0, 1]
    if prob >= THRESHOLD:
        st.error(f"HIGH FLOOD RISK TOMORROW — Probability: {prob:.1%} (Alert Threshold: {THRESHOLD:.0%})")
    else:
        st.success(f"LOW FLOOD RISK TOMORROW — Probability: {prob:.1%} (Alert Threshold: {THRESHOLD:.0%})")

chart_path = BASE_DIR / "flood_evaluation_report.png"
if chart_path.exists():
    st.divider()
    st.subheader("Model Validation Report (Chronological Holdout)")
    st.image(str(chart_path), use_container_width=True)
