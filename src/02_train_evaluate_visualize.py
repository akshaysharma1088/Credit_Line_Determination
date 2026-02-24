## 2) Python code: train model + evaluate + generate visualizations (AUC, PR, calibration, lift)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from xgboost import XGBClassifier

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "default_6m"
DROP_COLS = ["acct_id", "as_of_date"]

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300)
    plt.close()

df = pd.read_csv("cli_modeling_table.csv")
X = df.drop(columns=DROP_COLS + [TARGET])
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train XGBoost (strong default)
model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

p_test_raw = model.predict_proba(X_test)[:, 1]

# Probability calibration (isotonic on a calibration split)
X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
model.fit(X_tr, y_tr)
p_cal = model.predict_proba(X_cal)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_cal, y_cal)

p_test = iso.transform(model.predict_proba(X_test)[:, 1])

# Metrics
auc = roc_auc_score(y_test, p_test)
ap = average_precision_score(y_test, p_test)
print("AUC:", round(auc, 4), "PR-AUC:", round(ap, 4))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, p_test)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
savefig("Fig_ROC")

# PR Curve
prec, rec, _ = precision_recall_curve(y_test, p_test)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
savefig("Fig_PR")

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1])
plt.xlabel("Predicted PD")
plt.ylabel("Observed Default Rate")
plt.title("Calibration Plot (Isotonic)")
savefig("Fig_Calibration")

# Lift chart (deciles)
tmp = pd.DataFrame({"y": y_test.values, "p": p_test})
tmp["decile"] = pd.qcut(tmp["p"], 10, labels=False, duplicates="drop")
lift = tmp.groupby("decile").agg(obs_rate=("y", "mean"), avg_p=("p", "mean"), n=("y", "size")).reset_index()
lift = lift.sort_values("decile", ascending=False)

plt.figure()
plt.plot(range(1, len(lift) + 1), lift["obs_rate"].values, marker="o")
plt.xlabel("Score Decile (1=highest risk)")
plt.ylabel("Observed Default Rate")
plt.title("Lift by Risk Decile")
savefig("Fig_Lift_Deciles")

print(f"Saved figures to: {OUT_DIR}")
