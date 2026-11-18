---
title: Dhaka Flood Risk Predictor
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🌊 Dhaka Next-Day Flood Risk Predictor

An end-to-end machine learning pipeline that forecasts **next-day flood risk** in Dhaka, Bangladesh using 10 years of daily NASA POWER meteorological data, hydrological feature engineering, class-imbalanced classification, and an interactive Gradio web interface.

🔗 **Live Demo:** [Try the Predictor on Hugging Face Spaces](https://huggingface.co/spaces/Nuru824/dhaka-flood-predictor)

---

## 📌 Problem & Domain Framing

Urban and monsoon flooding in Dhaka is rarely triggered by a single isolated rainstorm—it occurs when **sustained multi-day rainfall falls on already saturated soil**. 

To model this physical mechanism without future data leakage:
1. **Hydrological Target Definition:** A day is labeled as a flood-risk day when **7-day cumulative rainfall exceeds 120 mm** (`rain_7day > 120`) **AND** net soil **moisture deficit is positive** (`moisture_deficit > 0`), yielding a natural class imbalance of **14.2% flood days**.
2. **Leakage-Free Forecasting (`shift(-1)`):** Today's meteorological features are aligned to predict **tomorrow's** flood state (`flood_tomorrow`), ensuring zero future information leaks into training.
3. **Cost-Sensitive Threshold Tuning:** In disaster early-warning systems, a **false negative (missed flood)** is far more costly than a **false positive (false alarm)**. Rather than optimizing raw accuracy, the model prioritizes **Recall** and **PR-AUC** using balanced class weights and a tuned decision threshold (`0.30`).

---

## 🏗️ Pipeline Architecture

```text
nasa_power_dhaka_10yr.csv
          │
          ▼
┌───────────────────────────┐
│  feature_engineering.py   │  Rolling rain (3d–30d), moisture deficit,
└─────────────┬─────────────┘  rain anomalies & cyclical seasonality
              │
              ▼
┌───────────────────────────┐
│         train.py          │  Chronological 80/20 split, Logistic Regression
└─────────────┬─────────────┘  baseline vs. balanced Random Forest
              │
              ▼
┌───────────────────────────┐
│        evaluate.py        │  PR-AUC, ROC-AUC, MCC, Brier Score,
└─────────────┬─────────────┘  threshold sensitivity & CI quality gate
              │
              ▼
┌───────────────────────────┐
│          app.py           │  Interactive Gradio UI (Hugging Face Spaces)
└───────────────────────────┘