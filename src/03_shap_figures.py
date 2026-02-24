# 3) Python code: SHAP plots (beeswarm, bar, waterfall, dependence)

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(42)

TARGET = "default_6m"
DROP_COLS = ["acct_id", "as_of_date"]

def save(name):
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

# Train model
model = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
    eval_metric="logloss", random_state=42
)

# Calibration split
X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
model.fit(X_tr, y_tr)
p_cal = model.predict_proba(X_cal)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_cal, y_cal)

# SHAP uses the underlying model; calibration is for policy/PD reporting.
# Use a representative sample for SHAP for speed.
X_shap = X_test.sample(n=min(20000, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_shap)

# Bar plot (global)
plt.figure()
shap.plots.bar(shap_values, max_display=20, show=False)
save("Fig_SHAP_Bar_Top20")

# Beeswarm (global distribution)
plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
save("Fig_SHAP_Beeswarm_Top20")

# Waterfall (single instance)
idx = 0
plt.figure()
shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
save("Fig_SHAP_Waterfall_Example")

# Dependence plots (pick key features if present)
candidates = ["pmt_to_bal_6m", "util_6m_slope", "score_delta_6m", "util_6m_mean"]
for feat in candidates:
    if feat in X_shap.columns:
        plt.figure()
        shap.plots.scatter(shap_values[:, feat], color=shap_values, show=False)
        save(f"Fig_SHAP_Dependence_{feat}")

print(f"Saved SHAP figures to: {OUT_DIR}")
