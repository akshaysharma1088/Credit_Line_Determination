"""Replication on real revolving-credit accounts (Section 7.6): the UCI 'Default of Credit Card
Clients' data set (Yeh and Lien 2009; 30,000 Taiwanese card accounts, April to September 2005).

Place the file at data/uci_credit_card.csv (scripts/get_card_data.sh). Sex, age and marital status
are never model features; sex and age are used only to measure and control approval gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from .common import SEED, DATA, ece, save
from .model import fit_lr, fit_xgb
from . import alternatives
from .robustness import evaluate, table, fast_auc, N_SEEDS

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{k}" for k in range(1, 7)]
PAID = [f"PAY_AMT{k}" for k in range(1, 7)]


def load():
    df = pd.read_csv(DATA / "uci_credit_card.csv")
    df.columns = [c.strip().replace(".", " ") for c in df.columns]
    y = df["default payment next month"].astype(int).values
    slog = lambda v: np.sign(v) * np.log1p(np.abs(v))
    X = pd.DataFrame({"limit": np.log(df["LIMIT_BAL"])})
    for e in (1, 2, 3):
        X[f"edu_{e}"] = (df["EDUCATION"] == e).astype(float)
    for c in PAY:
        X[c] = df[c].astype(float)
    for c in BILL + PAID:
        X[c] = slog(df[c].astype(float))
    util = df[BILL].div(df["LIMIT_BAL"], axis=0).clip(-0.5, 2.0)
    X["util_last"] = util["BILL_AMT1"]; X["util_mean"] = util.mean(1); X["util_slope"] = (util["BILL_AMT1"] - util["BILL_AMT6"]) / 5
    ratio = (df[PAID[:5]].values / np.clip(df[BILL[1:]].values, 1, None)).clip(0, 2)
    X["pay_ratio_mean"] = ratio.mean(1)
    X["max_delay"] = df[PAY].max(1).astype(float); X["months_late"] = (df[PAY] > 0).sum(1).astype(float)
    groups = {"sex": (df["SEX"] == 1).astype(int).values,          # 1 = male
              "age": (df["AGE"] < 30).astype(int).values}           # 1 = under 30
    return X, y, groups


CANDIDATES = {"drop limit": ["limit", "util_last", "util_mean", "util_slope"], "drop education": ["edu_1", "edu_2", "edu_3"],
              "drop repayment status": PAY + ["max_delay", "months_late"], "drop bill amounts": BILL,
              "drop payment amounts": PAID + ["pay_ratio_mean"], "drop utilization": ["util_last", "util_mean", "util_slope"],
              "drop latest status": ["PAY_0"], "logistic model": "__lr__"}


def main(n=N_SEEDS):
    X, y, groups = load()
    rows = {g: [] for g in groups}; alt = {g: [] for g in groups}; picks = {g: {} for g in groups}; reg = {g: [] for g in groups}
    for i in range(n):
        seed = SEED + i; idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
        lr = fit_lr(X.iloc[tr], y[tr]); gb = fit_xgb(X.iloc[tr], y[tr], X.iloc[va], y[va], seed=seed)
        raw_va, raw_te = gb.predict_proba(X.iloc[va])[:, 1], gb.predict_proba(X.iloc[te])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, y[va])
        base = {"yva": y[va], "yte": y[te], "raw_va": raw_va, "raw_te": raw_te, "cal_va": iso.predict(raw_va),
                "cal_te": iso.predict(raw_te), "lr_te": lr.predict_proba(X.iloc[te])[:, 1]}
        for g, A in groups.items():
            # code the group with the lower single-threshold approval rate on validation as A = 1
            q = np.quantile(raw_va, 0.5); low = (raw_va <= q)
            flip = low[A[va] == 1].mean() > low[A[va] == 0].mean()
            Ag = 1 - A if flip else A
            S = {**base, "Ava": Ag[va], "Ate": Ag[te]}
            r = evaluate(S); r["base_rate_A0"] = float(y[te][Ag[te] == 0].mean()); r["base_rate_A1"] = float(y[te][Ag[te] == 1].mean())
            r["share_A1"] = float(Ag[te].mean()); r["flipped"] = float(flip)
            rows[g].append(r)
            a = alternatives.search(X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], y[te], Ag[va], Ag[te], CANDIDATES, seed)
            picks[g][a["selected"]] = picks[g].get(a["selected"], 0) + 1
            alt[g].append({k: a[k] for k in ("gap_full", "gap_selected", "gap_reduction_pct", "profit_pct", "realized_loss_pct")})
            reg[g].append(alternatives.regularized(X.iloc[tr], y[tr], Ag[tr], X.iloc[va], y[va], Ag[va], X.iloc[te], y[te], Ag[te], seed))
        print("split", i, flush=True)
    save("16_card_data", {"n": int(len(y)), "default_rate": float(y.mean()), "n_splits": n,
                          "share_male": float(groups["sex"].mean()), "share_under_30": float(groups["age"].mean()),
                          "default_rate_male": float(y[groups["sex"] == 1].mean()), "default_rate_female": float(y[groups["sex"] == 0].mean()),
                          "default_rate_under_30": float(y[groups["age"] == 1].mean()), "default_rate_30_plus": float(y[groups["age"] == 0].mean()),
                          **{g: table(rows[g]) for g in groups},
                          "group_blind": {g: {"summary": table(alt[g]), "selected_counts": picks[g], "regularized": table(reg[g])} for g in groups}})


if __name__ == "__main__":
    main()
