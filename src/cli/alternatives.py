"""Group-blind alternatives (Section 7.9).

A lender that may not use the protected attribute in the decision can still test a single-threshold
policy for an approval gap and search for a less discriminatory alternative. We search over
group-blind candidates: the same boosted model retrained without one feature or one feature group,
and the logistic model. Every candidate uses ONE threshold for all accounts, set on validation so
that its expected loss fits the same budget. The protected attribute is used only to measure the gap.

Selection rule (on validation): among candidates whose expected profit is within TOL of the full
model's, take the one with the smallest approval gap. Reported on the test fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .common import GAMMA, LGD, EAD_INC, EL_BUDGET_SHARE
from .model import fit_lr, fit_xgb

TOL = 0.02


def _single(cal_va, score_va, score_te, budget):
    """Largest approval rate on validation whose expected loss fits the budget; returns masks."""
    order = np.argsort(score_va)
    cum = np.cumsum(cal_va[order] * LGD * EAD_INC)
    k = int(np.searchsorted(cum, budget, side="right"))
    if k == 0:
        return np.zeros(len(score_va), bool), np.zeros(len(score_te), bool)
    tau = score_va[order[k - 1]]
    return score_va <= tau, score_te <= tau


def _stats(y, a, A, cal):
    rev = GAMMA * EAD_INC * ((y == 0) & a).sum(); loss = LGD * EAD_INC * ((y == 1) & a).sum()
    return {"gap": float(a[A == 0].mean() - a[A == 1].mean()), "profit": float(rev - loss),
            "exp_profit": float(((GAMMA - cal[a] * LGD) * EAD_INC).sum()), "realized_loss": float(loss),
            "approval": float(a.mean())}


def search(Xtr, ytr, Xva, yva, Xte, yte, Ava, Ate, candidates, seed):
    """candidates: dict name -> list of columns to drop ('__lr__' = logistic model on all columns)."""
    out = {}; budget = None
    for name, drop in [("full", [])] + list(candidates.items()):
        if drop == "__lr__":
            m = fit_lr(Xtr, ytr); sva, ste = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        else:
            keep = [c for c in Xtr.columns if c not in drop]
            m = fit_xgb(Xtr[keep], ytr, Xva[keep], yva, seed=seed)
            sva, ste = m.predict_proba(Xva[keep])[:, 1], m.predict_proba(Xte[keep])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(sva, yva)
        cva, cte = iso.predict(sva), iso.predict(ste)
        if budget is None:
            budget = EL_BUDGET_SHARE * (cva[cva < GAMMA / LGD] * LGD * EAD_INC).sum()
        ava, ate = _single(cva, sva, ste, budget)
        out[name] = {"val": _stats(yva, ava, Ava, cva), "test": _stats(yte, ate, Ate, cte)}
    full = out["full"]
    ok = [n for n in out if out[n]["val"]["exp_profit"] >= (1 - TOL) * full["val"]["exp_profit"]]
    pick = min(ok, key=lambda n: abs(out[n]["val"]["gap"]))
    t, f = out[pick]["test"], full["test"]
    return {"selected": pick, "gap_full": f["gap"], "gap_selected": t["gap"],
            "gap_reduction_pct": 100 * (1 - t["gap"] / f["gap"]) if f["gap"] else 0.0,
            "profit_pct": 100 * (t["profit"] / f["profit"] - 1),
            "realized_loss_pct": 100 * (t["realized_loss"] / f["realized_loss"] - 1),
            "candidates": {n: {"gap": out[n]["test"]["gap"], "profit_pct": 100 * (out[n]["test"]["profit"] / f["profit"] - 1)}
                           for n in out}}


# ---------------------------------------------------------------- fairness-regularized model
LAMBDAS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]


def _fit_regularized(Xtr, ytr, Atr, lam, seed, rounds, Xva=None, yva=None):
    """Gradient boosting with a penalty on the between-group difference in mean predicted PD.

    Objective: class-weighted log-loss + (lam / 2) * n * (mean_p[A=1] - mean_p[A=0])^2.
    The protected attribute enters the training loss only. It is not a feature, and the fitted
    model scores every account from the same inputs, so one threshold can be applied to all.
    lam = 0 is the ordinary model. With Xva given, rounds are chosen by early stopping (lam = 0 only).
    """
    import xgboost as xgb
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    w = np.where(ytr == 1, spw, 1.0)
    n = len(ytr); n1 = Atr.sum(); n0 = n - n1
    s = np.where(Atr == 1, n / n1, -n / n0)

    def obj(margin, dtrain):
        p = 1.0 / (1.0 + np.exp(-margin))
        diff = p[Atr == 1].mean() - p[Atr == 0].mean()
        grad = w * (p - ytr) + lam * diff * s * p * (1 - p)
        hess = np.maximum(w * p * (1 - p), 1e-6)
        return grad, hess

    params = {"max_depth": 5, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
              "tree_method": "hist", "seed": seed, "nthread": 2, "base_score": 0.5}
    dtr = xgb.DMatrix(Xtr, label=ytr)
    if Xva is not None:
        dva = xgb.DMatrix(Xva, label=yva)

        def metric(margin, d):
            return "auc", rankdata_auc(yva, margin)
        bst = xgb.train(params, dtr, 600, obj=obj, evals=[(dva, "va")], custom_metric=metric, maximize=True,
                        early_stopping_rounds=50, verbose_eval=False)
        return bst, bst.best_iteration + 1
    return xgb.train(params, dtr, rounds, obj=obj, verbose_eval=False), rounds


def regularized(Xtr, ytr, Atr, Xva, yva, Ava, Xte, yte, Ate, seed, lams=LAMBDAS):
    """Single-threshold policies from fairness-regularized models, same expected-loss budget."""
    import xgboost as xgb
    out = {}; budget = None; rounds = None
    for lam in lams:
        if lam == 0.0:
            bst, rounds = _fit_regularized(Xtr, ytr, Atr, 0.0, seed, None, Xva, yva)
        else:
            bst, _ = _fit_regularized(Xtr, ytr, Atr, lam, seed, rounds)
        it = (0, rounds)
        sva = bst.predict(xgb.DMatrix(Xva), iteration_range=it); ste = bst.predict(xgb.DMatrix(Xte), iteration_range=it)
        iso = IsotonicRegression(out_of_bounds="clip").fit(sva, yva)
        cva, cte = iso.predict(sva), iso.predict(ste)
        if budget is None:
            budget = EL_BUDGET_SHARE * (cva[cva < GAMMA / LGD] * LGD * EAD_INC).sum()
        ava, ate = _single(cva, sva, ste, budget)
        r = rankdata_auc(yte, ste)
        out[lam] = {"val": _stats(yva, ava, Ava, cva), "test": _stats(yte, ate, Ate, cte), "auc": r}
    full = out[0.0]
    ok = [l for l in out if out[l]["val"]["exp_profit"] >= (1 - TOL) * full["val"]["exp_profit"]]
    pick = min(ok, key=lambda l: abs(out[l]["val"]["gap"]))
    res = {"selected_lambda": pick}
    for l in out:
        t, f = out[l]["test"], full["test"]
        res[f"gap_lam_{l:g}"] = t["gap"]; res[f"profit_pct_lam_{l:g}"] = 100 * (t["profit"] / f["profit"] - 1)
        res[f"loss_pct_lam_{l:g}"] = 100 * (t["realized_loss"] / f["realized_loss"] - 1); res[f"auc_lam_{l:g}"] = out[l]["auc"]
    t = out[pick]["test"]
    res.update({"gap_selected": t["gap"], "profit_pct_selected": 100 * (t["profit"] / full["test"]["profit"] - 1),
                "gap_reduction_pct_selected": 100 * (1 - t["gap"] / full["test"]["gap"]) if full["test"]["gap"] else 0.0})
    return res


def rankdata_auc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
