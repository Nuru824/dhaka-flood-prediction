import joblib
import pandas as pd
import gradio as gr

# Load saved model artifact
artifact = joblib.load("dhaka_flood_model.joblib")
model = artifact["rf_model"]
FEATURES = artifact["features"]
THRESHOLD = artifact["threshold"]

def predict_flood(
    prectotcorr, rain_3day, rain_7day, rain_14day, rain_30day, 
    moisture_deficit, rain_anomaly, is_monsoon, day_of_year,
    month_sin, month_cos, rh_3day_avg, rh_trend, temp_3day_avg
):
    input_df = pd.DataFrame([[
        prectotcorr, rain_3day, rain_7day, rain_14day, rain_30day,
        moisture_deficit, rain_anomaly, int(is_monsoon), day_of_year,
        month_sin, month_cos, rh_3day_avg, rh_trend, temp_3day_avg
    ]], columns=FEATURES)
    
    prob = model.predict_proba(input_df)[0, 1]
    alert = "HIGH FLOOD RISK TOMORROW" if prob >= THRESHOLD else "LOW FLOOD RISK TOMORROW"
    
    return (
        f"{alert} (Probability: {prob:.1%}  | Alert Threshold: {THRESHOLD:.0%})"
    )
    
demo = gr.Interface(
    fn=predict_flood,
    inputs=[
        gr.Number(label="Today's Rainfall (PRECTOTCORR, mm)", value=25.0),
        gr.Number(label="3-Day Cumulative Rain (mm)", value=65.0),
        gr.Number(label="7-Day Cumulative Rain (mm)", value=140.0),
        gr.Number(label="14-Day Cumulative Rain (mm)", value=240.0),
        gr.Number(label="30-Day Cumulative Rain (mm)", value=450.0),
        gr.Number(label="Moisture Deficit", value=15.0),
        gr.Number(label="Rain Anomaly", value=12.0),
        gr.Checkbox(label="Is Monsoon Season?", value=True),
        gr.Slider(1, 365, value=200, step=1, label="Day of Year"),
        gr.Number(label="Month Sin", value=-0.5),
        gr.Number(label="Month Cos", value=-0.86),
        gr.Number(label="3-Day Avg Relative Humidity (%)", value=88.0),
        gr.Number(label="Relative Humidity Trend", value=1.5),
        gr.Number(label="3-Day Avg Temperature (°C)", value=28.5),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Dhaka Tomorrow Flood Risk Predictor",
    description="Random Forest Classifier predicting next day flood risk from cumulative rainfall," 
        "soil moisture deficit, and seasonal meteorological features.",
    examples=[
        # Heavy monsoon flood scenario
        [45.0, 110.0, 185.0, 560.0, 35.0, 28.0, True, 210, -0.5, -0.86, 92.0, 2.5, 27.5],
        # Dry season low risk scenario
        [0.0, 0.0, 2.0, 5.0, 12.0, -40.0, -15.0, False, 30, 0.5, 0.86, 62.0, -1.0, 21.0],
    ],
)

if __name__ == '__main__':
    demo.launch()