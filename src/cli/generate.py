"""Synthetic credit-card portfolio generator (Section 4.1).

A latent financial-stress factor and an unobserved idiosyncratic factor drive
both behavior and default. Nineteen observed features are noisy proxies of
stress across four feature families plus account-master fields. A binary
protected-attribute proxy A is correlated with stress, limit and score. Default
comes from a nonlinear risk index dominated by threshold and interaction terms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import SEED, DATA, save, r

TARGET_DEFAULT_RATE = 0.18


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n=60_000, seed=SEED, w_unobs=0.90, nl_scale=1.0, group_shift=0.35, proxy_scale=1.0,
             keep_truth=False):
    """Defaults reproduce the paper's portfolio exactly. The keyword arguments are the
    generator assumptions varied in the sensitivity analysis (cli.robustness):
    w_unobs      weight of the unobserved factor in the risk index (caps discrimination)
    nl_scale     multiplier on the threshold and interaction terms (nonlinear structure)
    group_shift  mean stress shift of the disadvantaged group (true risk difference)
    proxy_scale  multiplier on the direct dependence of limit and score on A (proxy bias)
    """
    rng = np.random.default_rng(seed)
    year = rng.choice([1, 2, 3, 4], size=n)
    A = (rng.random(n) < 0.40).astype(int)                 # 1 = disadvantaged group
    stress = rng.normal(group_shift * A + 0.04 * (year - 1), 1.0, n)
    unobs = rng.normal(0, 1, n)                           # never observed by any model
    noise = lambda s: rng.normal(0, s, n)

    # account master
    tenure = np.clip(rng.gamma(3.0, 18.0, n) - 4 * stress, 3, 360).round()
    bureau = np.clip(705 - 38 * stress - (14 * proxy_scale) * A + noise(30), 450, 850).round()
    limit = np.clip(np.exp(8.6 - 0.18 * stress - (0.20 * proxy_scale) * A + noise(0.45)), 500, 50_000).round(-1)
    ext_util = np.clip(0.30 + 0.10 * stress + noise(0.12), 0, 1.2)

    # utilization dynamics
    util_6m = np.clip(0.38 + 0.14 * stress + noise(0.12), 0, 1.3)
    util_slope = 0.03 * stress + noise(0.05)
    util_3m = np.clip(util_6m + 3 * util_slope + noise(0.05), 0, 1.4)
    util_vol = np.clip(0.06 + 0.025 * stress + np.abs(noise(0.03)), 0, 0.6)

    # payment stability
    pmt_to_bal = np.clip(np.exp(-0.15 - 0.35 * stress + noise(0.35)), 0.02, 4.0)
    consec_ontime = np.clip(rng.poisson(np.clip(14 - 4 * stress, 0.5, None)), 0, 36)
    full_to_min = np.clip(np.exp(0.7 - 0.30 * stress + noise(0.5)), 0.3, 12)
    pmt_timing_dev = np.clip(2.5 + 1.6 * stress + noise(2.0), -5, 20)

    # behavioral stability
    spend_cv = np.clip(0.35 + 0.05 * stress + noise(0.12), 0.02, 2.0)
    revolver = np.clip(0.45 + 0.08 * stress + noise(0.2), 0, 1)
    spend_entropy = np.clip(2.2 - 0.05 * stress + noise(0.35), 0.1, 4)

    # risk drift
    score_d6 = np.round(-9 * stress + noise(14))
    score_d3 = np.round(0.55 * score_d6 + noise(8))
    inquiries = rng.poisson(np.clip(0.8 + 0.5 * stress, 0.05, None))
    delinq_spike = (rng.random(n) < _sigmoid(-3.2 + 0.9 * stress)).astype(int)

    # nonlinear risk index: threshold and interaction terms on observed behavior
    risk = (0.25 * stress + w_unobs * unobs
            + (1.6 * nl_scale) * ((util_3m > 0.70) & (pmt_to_bal < 0.45))
            + (1.2 * nl_scale) * (util_slope > 0.05)
            + (1.0 * nl_scale) * (consec_ontime < 6)
            + (1.0 * nl_scale) * delinq_spike * (util_3m > 0.6)
            + (0.7 * nl_scale) * (score_d6 < -20)
            + (0.5 * nl_scale) * (inquiries >= 3))
    lo, hi = -10.0, 10.0  # intercept by bisection to hit the target base rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(risk + mid).mean() > TARGET_DEFAULT_RATE:
            hi = mid
        else:
            lo = mid
    pd_true = _sigmoid(risk + mid)
    default = (rng.random(n) < pd_true).astype(int)

    df = pd.DataFrame({
        "account_id": np.arange(1, n + 1), "year": year, "A": A,
        "credit_limit": limit, "tenure_months": tenure, "bureau_score": bureau, "ext_util": ext_util,
        "util_3m": util_3m, "util_6m_mean": util_6m, "util_slope_6m": util_slope, "util_vol_6m": util_vol,
        "pmt_to_bal_6m": pmt_to_bal, "consec_ontime": consec_ontime, "full_to_min_ratio": full_to_min,
        "pmt_timing_dev": pmt_timing_dev,
        "spend_cv": spend_cv, "revolver_consistency": revolver, "spend_entropy": spend_entropy,
        "score_delta_3m": score_d3, "score_delta_6m": score_d6, "inquiries_6m": inquiries,
        "delinq_spike": delinq_spike,
        "default": default,
    })
    if keep_truth:
        df["pd_true"] = pd_true
    return df


def main():
    df = generate()
    df.to_csv(DATA / "synthetic_portfolio.csv", index=False)
    summary = {
        "n_accounts": int(len(df)),
        "default_rate": r(df["default"].mean(), 4),
        "default_rate_A0": r(df.loc[df.A == 0, "default"].mean(), 4),
        "default_rate_A1": r(df.loc[df.A == 1, "default"].mean(), 4),
        "share_A1": r(df["A"].mean(), 4),
        "n_by_split": {"train_years_1_2": int((df.year <= 2).sum()),
                       "validation_year_3": int((df.year == 3).sum()),
                       "test_year_4": int((df.year == 4).sum())},
    }
    save("01_data_summary", summary)
    print(summary)


if __name__ == "__main__":
    main()
