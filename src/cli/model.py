"""Modeling, calibration, explainability, ablation, economics and fairness
(Sections 5, 6 and 7.1 to 7.5). Run after cli.generate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (SEED, DATA, FIGS, FEATURES, FAMILIES, GAMMA, LGD, EAD_INC,
                     BREAK_EVEN_PD, EL_BUDGET_SHARE, disc_metrics, ece, save, r)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def split(df):
    return df[df.year <= 2], df[df.year == 3], df[df.year == 4]


def fit_xgb(Xtr, ytr, Xva, yva, seed=SEED):
    # Early stopping monitors validation AUC. Monitoring unweighted log-loss while training with
    # scale_pos_weight (as in the first release) never triggers a stop and leaves the model over-trained.
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    m = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=600, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                          early_stopping_rounds=50, eval_metric="auc", random_state=seed,
                          n_jobs=2, tree_method="hist")
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_lr(Xtr, ytr):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, ytr)
    return m


# ---------- economics helpers ----------
def expected_loss(pd_hat):
    return pd_hat * LGD * EAD_INC


def threshold_for_budget(pd_val, budget):
    """Largest tau such that the expected loss of approving PD < tau stays within budget."""
    order = np.sort(pd_val)
    cum = np.cumsum(expected_loss(order))
    k = np.searchsorted(cum, budget, side="right")
    return float(order[k - 1]) if k > 0 else 0.0


def realized(y, approve):
    """Realized revenue, loss and net profit of approved accounts (simulation units)."""
    a = approve.astype(bool)
    revenue = GAMMA * EAD_INC * ((y == 0) & a).sum()
    loss = LGD * EAD_INC * ((y == 1) & a).sum()
    return {"approval_rate": float(a.mean()), "approved_default_rate": float(y[a].mean()) if a.any() else 0.0,
            "revenue": float(revenue), "loss": float(loss), "net_profit": float(revenue - loss)}


def approve_at_rate(score, rate):
    tau = np.quantile(score, rate)
    return score <= tau


def fairness(y, approve, A):
    a = approve.astype(bool); out = {}
    p0, p1 = a[A == 0].mean(), a[A == 1].mean()
    tpr0, tpr1 = a[(A == 0) & (y == 0)].mean(), a[(A == 1) & (y == 0)].mean()   # approval among non-defaulters
    fpr0, fpr1 = a[(A == 0) & (y == 1)].mean(), a[(A == 1) & (y == 1)].mean()   # approval among defaulters
    out["approval_A0"], out["approval_A1"] = float(p0), float(p1)
    out["dp_diff"] = float(abs(p0 - p1))
    out["tpr_gap"] = float(abs(tpr0 - tpr1))
    out["fpr_gap"] = float(abs(fpr0 - fpr1))
    out["di_ratio"] = float(p1 / p0) if p0 > 0 else float("nan")
    return out


def main():
    df = pd.read_csv(DATA / "synthetic_portfolio.csv")
    tr, va, te = split(df)
    Xtr, Xva, Xte = tr[FEATURES], va[FEATURES], te[FEATURES]
    ytr, yva, yte = tr["default"].values, va["default"].values, te["default"].values
    Ate, Ava = te["A"].values, va["A"].values

    # ---- models ----
    lr = fit_lr(Xtr, ytr)
    gb = fit_xgb(Xtr, ytr, Xva, yva)
    p_lr_va, p_lr_te = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]
    raw_va, raw_te = gb.predict_proba(Xva)[:, 1], gb.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, yva)
    cal_va, cal_te = iso.predict(raw_va), iso.predict(raw_te)

    disc = {"logistic": disc_metrics(yte, p_lr_te), "gb_raw": disc_metrics(yte, raw_te),
            "gb_isotonic": disc_metrics(yte, cal_te), "best_iteration": int(gb.best_iteration),
            "split_sizes": {"train": len(tr), "validation": len(va), "test": len(te)}}
    save("02_discrimination_calibration", disc)

    # ROC
    fig, ax = plt.subplots(figsize=(4, 3.2))
    for name, p in [("Logistic", p_lr_te), ("XGBoost + isotonic", cal_te)]:
        f, t, _ = roc_curve(yte, p); ax.plot(f, t, label=f"{name} (AUC={roc_auc_score(yte, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC, out-of-time test fold (year 4)"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "fig2_roc.png", dpi=300); plt.close(fig)

    # reliability
    fig, ax = plt.subplots(figsize=(4, 3.2))
    edges = np.linspace(0, 1, 11)
    for name, p, mk in [("Uncalibrated", raw_te, "o"), ("Isotonic-calibrated", cal_te, "s")]:
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
        ys = [yte[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
        ax.plot(xs, ys, marker=mk, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("Predicted PD"); ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability diagram"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_reliability.png", dpi=300); plt.close(fig)

    # ---- SHAP ----
    sample = Xte.sample(5000, random_state=SEED)
    sv = shap.TreeExplainer(gb).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    save("03_shap_importance", {k: r(v, 4) for k, v in imp.items()})
    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.barh(top.index, top.values, color="#2F6DB5"); ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature importance"); fig.tight_layout(); fig.savefig(FIGS / "fig4_shap.png", dpi=300); plt.close(fig)

    # ---- ablation ----
    base_auc = roc_auc_score(yte, raw_te); abl = {}
    for fam, cols in FAMILIES.items():
        keep = [c for c in FEATURES if c not in cols]
        m = fit_xgb(tr[keep], ytr, va[keep], yva)
        abl[fam] = r(base_auc - roc_auc_score(yte, m.predict_proba(te[keep])[:, 1]), 4)
    save("04_ablation", {"base_auc": r(base_auc, 4), "auc_drop": abl})

    # ---- economics (Section 7.4) ----
    profitable_va = cal_va < BREAK_EVEN_PD
    budget = EL_BUDGET_SHARE * expected_loss(cal_va[profitable_va]).sum()
    # single threshold: largest validation approval rate (ranked by raw score) whose expected
    # loss fits the budget; tau* is reported on the calibrated PD scale
    q_star = max(q for q in np.linspace(0.05, 0.95, 901)
                 if expected_loss(cal_va[raw_va <= np.quantile(raw_va, q)]).sum() <= budget)
    raw_tau = np.quantile(raw_va, q_star)
    tau = float(iso.predict([raw_tau])[0])
    appr_gb = raw_te <= raw_tau
    rate = appr_gb.mean()
    appr_lr = approve_at_rate(p_lr_te, rate)
    e_gb, e_lr = realized(yte, appr_gb), realized(yte, appr_lr)
    econ = {"break_even_pd": BREAK_EVEN_PD, "tau_star": r(tau, 4), "approval_rate": r(rate, 4),
            "logistic": e_lr, "gradient_boosting": e_gb,
            "change": {"approved_default_rate_pp": r(100 * (e_gb["approved_default_rate"] - e_lr["approved_default_rate"]), 2),
                       "loss_pct": r(100 * (e_gb["loss"] / e_lr["loss"] - 1), 1),
                       "net_profit_pct": r(100 * (e_gb["net_profit"] / e_lr["net_profit"] - 1), 1)}}
    save("05_economics", econ)
    fig, ax = plt.subplots(figsize=(4, 3.2)); x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, [e_lr["revenue"], e_lr["loss"], e_lr["net_profit"]], w, label="Logistic", color="#9AA5B1")
    ax.bar(x + w / 2, [e_gb["revenue"], e_gb["loss"], e_gb["net_profit"]], w, label="XGBoost", color="#2F6DB5")
    ax.set_xticks(x); ax.set_xticklabels(["Revenue", "Realized loss", "Net profit"]); ax.set_ylabel("Simulation units")
    ax.set_title(f"Matched approval rate ({100 * rate:.1f}%)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig5_economics.png", dpi=300); plt.close(fig)

    # ---- fairness (Section 7.5) ----
    groups = {}
    for g in (0, 1):
        m = Ate == g
        groups[f"A{g}"] = {"auc": r(roc_auc_score(yte[m], cal_te[m])), "ece": r(ece(yte[m], cal_te[m])),
                           "approval": r(appr_gb[m].mean(), 4), "approved_default_rate": r(yte[m & appr_gb].mean(), 4),
                           "base_default_rate": r(yte[m].mean(), 4)}
    before = fairness(yte, appr_gb, Ate)

    # group thresholds chosen on validation: equal approval rates (ranked by the raw score, which
    # isotonic calibration preserves but ties), largest common rate whose expected loss fits the budget
    best = None
    for q in np.linspace(0.05, 0.95, 901):
        t0g = np.quantile(raw_va[Ava == 0], q); t1g = np.quantile(raw_va[Ava == 1], q)
        appr = np.where(Ava == 1, raw_va <= t1g, raw_va <= t0g)
        if expected_loss(cal_va[appr]).sum() <= budget:
            best = (t0g, t1g)
    t0, t1 = best
    appr_fair = np.where(Ate == 1, raw_te <= t1, raw_te <= t0)
    after = fairness(yte, appr_fair, Ate)
    e_fair = realized(yte, appr_fair)
    save("06_fairness", {"groups_single_threshold": groups, "before": before, "after": after,
                         "group_raw_score_thresholds": {"A0": r(t0, 4), "A1": r(t1, 4)},
                         "economics_single": e_gb, "economics_recalibrated": e_fair})

    # frontier: hold A0 at the single threshold, extend access to A1 in order of raw score
    pts = []
    r0 = appr_gb[Ate == 0].mean(); r1 = appr_gb[Ate == 1].mean()
    for q in np.linspace(r1, r0, 25):
        t = np.quantile(raw_te[Ate == 1], q)
        appr = np.where(Ate == 1, raw_te <= t, appr_gb)
        f = fairness(yte, appr, Ate); e = realized(yte, appr)
        pts.append({"approval_A1": r(f["approval_A1"], 4), "dp_diff": r(f["dp_diff"], 4),
                    "signed_gap": r(f["approval_A0"] - f["approval_A1"], 4), "net_profit": r(e["net_profit"], 1)})
    pts = [p for p in pts if p["signed_gap"] >= -0.005]
    save("07_frontier", pts)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.plot([p["signed_gap"] for p in pts], [p["net_profit"] / 1e6 for p in pts], "o-", ms=3, color="#2F6DB5")
    ax.invert_xaxis(); ax.set_xlabel("Demographic-parity gap (approval A0 minus A1)"); ax.set_ylabel("Realized net profit (millions)")
    ax.set_title("Access expansion path (larger risk budget)"); fig.tight_layout(); fig.savefig(FIGS / "fig6_expansion_path.png", dpi=300); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
