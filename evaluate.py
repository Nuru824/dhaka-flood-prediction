from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)

# 1: Load saved artifact and data
BASE_DIR = Path(__file__).parent
ARTIFACT_PATH = BASE_DIR/ "dhaka_flood_model.joblib"
DATA_PATH = BASE_DIR/ "dhaka_features_engineered.csv"

artifact = joblib.load(ARTIFACT_PATH)
rf_model = artifact["rf_model"]
lr_model = artifact["lr_model"]
scaler = artifact["scaler"]
FEATURES = artifact["features"]
THRESHOLD = artifact["threshold"]

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Recreate target label and shift to predict tomorrow's flood
df["flood"] = ((df["rain_7day"] > 120) & (df["moisture_deficit"] > 0)).astype(int)
df["flood_tomorrow"] = df["flood"].shift(-1)
df = df.dropna().reset_index(drop=True)
df["flood_tomorrow"] = df["flood_tomorrow"].astype(int)

# Use the exact same 80/20 chronological holdout split
split_idx = int(len(df) * 0.8)
X_test = df[FEATURES].iloc[split_idx:]
y_test = df["flood_tomorrow"].iloc[split_idx:]
test_dates = df["date"].iloc[split_idx:]

print(f"Evaluating on {len(X_test)} holdout days ({test_dates.min().date()} to {test_dates.max().date()})")
print(f"ACtual flood days in test set: {y_test.sum()} ({y_test.mean()* 100:.1f}%)\n")

# Generate predictions
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_pred = (rf_proba >= THRESHOLD).astype(int)

X_test_scaled = scaler.transform(X_test)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_pred = (lr_proba >= THRESHOLD).astype(int)

# 3. Core metrics report

def print_metrics(name:str, y_true: pd.Series, y_prob: np.ndarray, y_hat: np.ndarray) -> dict:
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_hat)
    rec = recall_score(y_true, y_hat, zero_division=0)
    prec = precision_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    
    print("=" * 55)
    print(f"  {name} (threshold = {THRESHOLD})")
    print("=" * 55)
    print(f"ROC-AUC          : {roc_auc:.3f}")
    print(f"PR_AUC (Avg Prec): {pr_auc:.3f}  <--- Key metric for 14.2% imbalance")
    print(f"Brier Score      : {brier: .4f} (lower = better probability calibration)")
    print(f"MCC              : {mcc:.3f} -1 to +1 balanced correlation")
    print(f"Recall (Flood)   : {rec:.3f}")
    print(f"Precision (Flood): {prec:.3f}")
    print(f"F1 score (Flood) : {f1:.3f}\n")
    print(classification_report(y_true, y_hat, target_names=["no_flood", "flood"]))
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    print(confusion_matrix(y_true, y_hat))
    print()
    
    return {"recall": rec, "pr_auc": pr_auc, "f1": f1}
lr_metrics = print_metrics("Logistic Regression Baseline", y_test, lr_proba, lr_pred)
rf_metrics = print_metrics("Random Forest Classfifier", y_test, rf_proba, rf_pred)

# 4. THRESHOLD SENNSITIVITY TABLE

print("Thershold sensitivity (Random Forest): ")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'False Alarms':>14} {'Missed Floods':>15}")
for thresh in [0.20, 0.25, 0.30, 0.40, 0.50]:
    preds = (rf_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)
    print(f"{thresh:>10.2f} {p:>10.3f} {r:>10.3f} {f:>10.3f} {fp:>14d} {fn:>15d}")
    
# 5. Save Diagnostic plots

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ROC CURVE
for name, proba in [("Logistic Reg", lr_proba), ("Random Forest", rf_proba)]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(rec, prec, label=name)
axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.4, label="Baseline")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall curve")
axes[1].legend()

# Top 10 Feature importances
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
importances.head(10).plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].invert_yaxis()
axes[2].set_title("Top 10 Feature Importances")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(BASE_DIR/ "flood_evaluation_report.png", dpi=150, bbox_inches="tight")
print("\nSaved evaluation chart to flood_evaluation_report.png")

# 6. Deployment Quality Gate
MIN_REQUIRED_RECALL = 0.75
if rf_metrics["recall"] < MIN_REQUIRED_RECALL:
    print(f"FAILED QUALITY GATE: Recall {rf_metrics['recall']:.3f} < {MIN_REQUIRED_RECALL}")
    sys.exit(1)
print(f"PASSED QUALITY GATE: Recall {rf_metrics['recall']:.3f} >= {MIN_REQUIRED_RECALL}")
