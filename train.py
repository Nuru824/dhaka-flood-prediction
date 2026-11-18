from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, RocCurveDisplay
    )
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------
# 1. LOAD ENGINEERED FEATURES
# ---------------------------------
file_dir = Path(__file__).parent
filename = file_dir / 'dhaka_features_engineered.csv'
df = pd.read_csv(filename,  parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True) # gives rows clean sequential number

# --------------------------------
# 2. CREATE THE FLOOD LABEL
# --------------------------------
# Strategy: a day if a flood day if:
#     - 7-day cumulative rainfall excceds 120mm (sustained heavy rain)
#     - AND mositure deficit > 0 (Soil already saturated)
RAIN_THRESHOLD = 120 # mm over 7 days
MOISTURE_THRESHOLD = 0 # net positive water balance

df['flood'] = (
    (df['rain_7day'] > RAIN_THRESHOLD) & 
    (df['moisture_deficit'] > MOISTURE_THRESHOLD)
).astype(int) # convert T/F to 1/0

print("Label distribution:")
print(df['flood'].value_counts())
print(f"Flood rate: {df['flood'].mean()*100:.1f}% of days\n")

# --------------------------------
# 3. SHIFT TARGET: PREDICT TOMORROW's FLOOD
# Critical step: we want to predict whether tomorrow will flood,
# using only today's information. So we shift the label back by 1.
# Without this shift you'd be leaking the future into training.
# --------------------------------

df['flood_tomorrow'] = df['flood'].shift(-1) # shift up by 1, so today's features predict tomorrow's flood
df = df.dropna().reset_index(drop=True) # drop last row which has NaN label after shift
df['flood_tomorrow'] = df['flood_tomorrow'].astype(int) # convert from float to int (0/1)

# --------------------------------
# 4. DEFINE FEATURES AND TARGET
# --------------------------------
FEATURES = [
    'PRECTOTCORR',    # today's raw rainfall
    'rain_3day',
    'rain_7day',
    'rain_14day',
    'rain_30day',
    'moisture_deficit',
    'rain_anomaly',
    'is_monsoon',
    'day_of_year',
    'month_sin',
    'month_cos',
    'rh_3day_avg',
    'rh_trend',
    'temp_3day_avg',
]
X = df[FEATURES]
y = df['flood_tomorrow']

# -------------------------------
# 5. TIME-SERIES SPLIT
# -------------------------------
tscv = TimeSeriesSplit(n_splits=5)
# For final evaluation, use the last 20% of data as holdout
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]    
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training samples : {len(X_train)}")
print(f"Testing samples : {len(X_test)}\n")

# -------------------------------
# 6. HANDLE CLASS IMBALANCE
#
# Floods are rare — maybe 10-20% of days. If we ignore this,
# the model learns to just predict "no flood" every day and
# gets 85% accuracy while being completely useless.
# class_weight='balanced' tells the model to penalize
# missing a flood much more than a false alarm.
# ─────────────────────────────────────────────
classes = np.array([0, 1])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0:weights[0], 1:weights[1]}
print(f"Class weights: {class_weight_dict}\n")

# ------------------------------
# 7. TRAIN MODELS
# ------------------------------

# --- Model A: Logistic Regression (basline) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

# --- Model B: Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)
rf_model.fit(X_train, y_train)

# ------------------------------
# 8. Evaluate
#
# Key metrics for imbalanced flood prediction:
#    Precision - of days we predicted flood, how many actually flooded?
#    Recall    - of actual flood days, how many did we catch
#    AUC-ROC   - overall ranking ability (0.5 = random, 1.0=perfect)

# For disaster prediction Recall matters more than precision -
# missing a flood is worse than a false alarm
def evaluate(name, model, X_t, y_t, scaled=False):
    if scaled:
        X_t = scaler.transform(X_t)
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]
    auc = roc_auc_score(y_t, y_proba)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"AUC-ROC: {auc:.3f}")
    print(classification_report(y_t, y_pred, target_names=['no_flood', 'flood']))
    print("Confusion_matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_t, y_pred))
    return y_proba
lr_proba = evaluate("Logistic Regression", lr_model, X_test, y_test, scaled=True)
rf_proba = evaluate("Random Forest", rf_model, X_test, y_test, scaled=False)

# -------------------------
# 9. FEATURE IMPORTANCE (Random Forest)
# -------------------------
importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("\nFeature importances (Random Forest):")
print(importances.round(4).to_string())

# ---------------------------------
# 10. Plots
# --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- ROC curves ---
ax = axes[0]

for name, proba in [("Logistic Reg", lr_proba), ("Random Forest", rf_proba)]:
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
ax.set_title("ROC curve")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
ax.legend()

# --- Precision-Recall curves ---
ax = axes[1]

for name, proba in [("Logistic Reg", lr_proba), ("Random Forest", rf_proba)]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, label=name)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve\n(more useful for imbalanced data)")
ax.legend()

# --- Feature importance ---
ax = axes[2]
importances.head(10).plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title("Top 10 Feature Importances\n(Random Forest)")
ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig('flood_classification_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to flood_classification_results.png")

# -----------------------------------
# 11. SAVE PREDICTIONS
# -----------------------------------
results = df.iloc[split_idx:][['date', 'flood_tomorrow']].copy()
results['rf_proba'] = rf_proba
results['rf_Pred'] = rf_model.predict(X_test)
results['lr_proba'] = lr_proba
results['lr_pred'] = lr_model.predict(X_test_scaled)
results.to_csv('flood_predictions.csv', index=False)
print("Predictions saved to flood_predictions.csv")

# ---------------------------------
# 12. THRESHOLD TUNING

# Default threshold is 0.5 - but for flood prediction you 
# might want to lower it (e.g. 0.3) to catch more floods 
# at the cost of more false alarms. Print a quick table
# ---------------------------------
print("\nThreshold sensitivity(Random Forest):")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")

for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
    preds = (rf_proba >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)
    print(f"{thresh:>10.1f} {p:>10.3f} {r:>10.3f} {f:>10.3f}")

import joblib

artifact = {
    "rf_model": rf_model,
    "lr_model": lr_model,
    "scaler": scaler,
    "features": FEATURES,
    "threshold": 0.3,
}
joblib.dump(artifact, "dhaka_flood_model.joblib")
print("Saved full evaluation artifact to dhaka_flood_model.joblib")
