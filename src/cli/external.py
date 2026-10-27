"""External validation on the real public credit_data benchmark (Section 7.6).

Source: Rdatasets archive, modeldata/credit_data.csv (4,454 applicants).
Download it once with scripts/get_credit_data.sh, or place the CSV in data/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from .common import SEED, DATA, disc_metrics, ece, save, r
from .model import fit_lr, fit_xgb, fairness


def load():
    df = pd.read_csv(DATA / "credit_data.csv").drop(columns=["rownames"], errors="ignore")
    y = (df.pop("Status") == "bad").astype(int).values
    age = df["Age"].values
    num = df.select_dtypes("number").columns
    cat = [c for c in df.columns if c not in num]
    df[num] = df[num].fillna(df[num].median())
    for c in cat:
        df[c] = df[c].fillna(df[c].mode()[0])
    X = pd.get_dummies(df, columns=cat, drop_first=True).astype(float)
    return X, y, age


def main():
    X, y, age = load()
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=SEED)
    lr = fit_lr(X.iloc[tr], y[tr])
    gb = fit_xgb(X.iloc[tr], y[tr], X.iloc[va], y[va])
    p_lr = lr.predict_proba(X.iloc[te])[:, 1]
    raw_va, raw_te = gb.predict_proba(X.iloc[va])[:, 1], gb.predict_proba(X.iloc[te])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, y[va])
    cal_te = iso.predict(raw_te)

    # approval policy: approve the validation share of good applicants, ranked by raw score
    rate = 1 - y[va].mean()
    t = np.quantile(raw_va, rate)
    A_te = (age[te] < 30).astype(int); A_va = (age[va] < 30).astype(int)
    single = raw_te <= t
    t0, t1 = np.quantile(raw_va[A_va == 0], rate), np.quantile(raw_va[A_va == 1], rate)
    recal = np.where(A_te == 1, raw_te <= t1, raw_te <= t0)
    out = {"n": int(len(y)), "bad_rate": r(y.mean(), 4), "split": {"train": len(tr), "validation": len(va), "test": len(te)},
           "logistic": disc_metrics(y[te], p_lr), "gb_raw": disc_metrics(y[te], raw_te), "gb_isotonic": disc_metrics(y[te], cal_te),
           "protected": "age < 30", "approval_rate_single": r(single.mean(), 4),
           "fairness_single": fairness(y[te], single, A_te), "fairness_recalibrated": fairness(y[te], recal, A_te)}
    save("08_external_credit_data", out)
    print({k: out[k] for k in ("n", "bad_rate", "approval_rate_single")})


if __name__ == "__main__":
    main()
