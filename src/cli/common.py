"""Shared configuration and metrics for the CLI reproducibility package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

SEED = 20261027
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
for d in (DATA, RESULTS, FIGS):
    d.mkdir(exist_ok=True)

# Economics (Section 3): standardized increment, utilization, margin, LGD.
INCREMENT = 1000.0      # offered increase, limit units
UTIL_INC = 0.5          # expected incremental utilization fraction U
GAMMA = 0.18            # net margin rate over the horizon
LGD = 0.80              # loss given default  -> break-even PD = GAMMA/LGD = 0.225
EAD_INC = INCREMENT * UTIL_INC
BREAK_EVEN_PD = GAMMA / LGD
EL_BUDGET_SHARE = 0.60  # EL target = 60% of the loss of approving every profitable account

FAMILIES = {
    "utilization": ["util_3m", "util_6m_mean", "util_slope_6m", "util_vol_6m"],
    "payment": ["pmt_to_bal_6m", "consec_ontime", "full_to_min_ratio", "pmt_timing_dev"],
    "behavioral": ["spend_cv", "revolver_consistency", "spend_entropy"],
    "risk_drift": ["score_delta_3m", "score_delta_6m", "inquiries_6m", "delinq_spike"],
}
MASTER = ["credit_limit", "tenure_months", "bureau_score", "ext_util"]
FEATURES = MASTER + [f for fam in FAMILIES.values() for f in fam]


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, bins=10):
    """Expected calibration error with equal-width probability bins."""
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def disc_metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "ks": ks_stat(y, p),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}


def save(name, obj):
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2)


def r(x, k=3):
    return float(round(float(x), k))
