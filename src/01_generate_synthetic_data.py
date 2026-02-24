##1) Python code: generate synthetic reference data (distribution-preserving)

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_synthetic_cli_data(n=200_000, as_of_date="2025-12-31"):
    # Account basics
    acct_id = np.arange(1, n + 1)
    tenure_months = RNG.integers(1, 240, size=n)

    # Credit score distribution (roughly tri-modal-ish)
    score_current = np.clip(
        (RNG.normal(680, 60, size=n) + RNG.normal(0, 25, size=n)),
        300, 850
    ).astype(int)

    # Limits correlated with score and tenure
    base_limit = 500 + (score_current - 300) * 30 + tenure_months * 15
    curr_limit = np.clip(base_limit + RNG.normal(0, 2000, size=n), 500, 50000)

    # Utilization patterns
    util_12m_mean = np.clip(RNG.beta(2.2, 6.5, size=n) + RNG.normal(0, 0.04, size=n), 0, 1.5)
    util_6m_mean = np.clip(util_12m_mean + RNG.normal(0, 0.06, size=n), 0, 1.5)
    util_3m_mean = np.clip(util_6m_mean + RNG.normal(0, 0.07, size=n), 0, 1.5)
    util_6m_slope = np.clip(util_3m_mean - util_12m_mean + RNG.normal(0, 0.03, size=n), -1.0, 1.0)
    util_vol_12m = np.clip(RNG.gamma(2.0, 0.05, size=n), 0, 0.8)

    # Spend (correlated with limit and utilization)
    spend_mean_6m = np.clip(curr_limit * util_6m_mean * RNG.uniform(0.05, 0.18, size=n), 50, 15000)
    spend_vol_6m = np.clip(spend_mean_6m * RNG.uniform(0.2, 1.3, size=n), 10, 20000)

    # Payment to balance ratio: higher is better
    pmt_to_bal_6m = np.clip(RNG.lognormal(mean=0.0, sigma=0.35, size=n), 0.05, 3.0)
    pmt_to_bal_3m = np.clip(pmt_to_bal_6m + RNG.normal(0, 0.15, size=n), 0.05, 3.0)

    # Revolver ratio: higher implies revolving behavior
    revolver_ratio_12m = np.clip(RNG.beta(2.5, 2.5, size=n), 0, 1)

    # On-time streak: impacted by score + utilization + payment ratio
    streak_base = (
        8
        + (score_current - 650) / 50
        - util_6m_mean * 3
        + (pmt_to_bal_6m - 1.0) * 4
        + RNG.normal(0, 2.0, size=n)
    )
    on_time_pmt_streak_12m = np.clip(np.round(streak_base), 0, 12).astype(int)

    # Inquiries and score drift
    inq_6m = np.clip(RNG.poisson(lam=1.0, size=n) + (util_6m_mean > 0.8).astype(int), 0, 12)
    score_delta_6m = np.clip(
        RNG.normal(0, 18, size=n) - (inq_6m * 2) - (util_6m_slope * 15),
        -150, 150
    ).astype(int)

    # Delinquency counts influenced by utilization, payment ratio, score
    dpd_risk = sigmoid(
        -2.5
        + util_6m_mean * 2.2
        - (pmt_to_bal_6m - 1.0) * 1.4
        - (score_current - 650) / 80
        + util_vol_12m * 1.0
    )
    dpd_30_12m = RNG.binomial(3, np.clip(dpd_risk, 0, 0.6), size=n)
    dpd_60_12m = RNG.binomial(2, np.clip(dpd_risk - 0.12, 0, 0.5), size=n)

    # Decision simulation fields
    proposed_cli = np.clip(curr_limit * RNG.uniform(0.05, 0.35, size=n), 100, 10000)
    expected_util_uplift = np.clip(RNG.normal(0.08, 0.04, size=n), 0.0, 0.25)
    lgd = np.clip(RNG.normal(0.72, 0.12, size=n), 0.2, 0.95)
    ead = np.clip(curr_limit * util_6m_mean, 0, None)

    # Label: default_6m (risk model “ground truth”)
    logit_pd = (
        -4.0
        + util_6m_mean * 1.8
        + np.maximum(util_6m_slope, 0) * 1.2
        + util_vol_12m * 0.8
        - (pmt_to_bal_6m - 1.0) * 1.4
        - (score_current - 650) / 90
        + (dpd_30_12m * 0.35)
        + (dpd_60_12m * 0.8)
        + (inq_6m * 0.05)
    )
    pd_true = sigmoid(logit_pd)
    default_6m = RNG.binomial(1, np.clip(pd_true, 0, 0.5), size=n)

    df = pd.DataFrame({
        "acct_id": acct_id,
        "as_of_date": as_of_date,
        "curr_limit": curr_limit,
        "tenure_months": tenure_months,
        "util_3m_mean": util_3m_mean,
        "util_6m_mean": util_6m_mean,
        "util_12m_mean": util_12m_mean,
        "util_6m_slope": util_6m_slope,
        "util_vol_12m": util_vol_12m,
        "pmt_to_bal_3m": pmt_to_bal_3m,
        "pmt_to_bal_6m": pmt_to_bal_6m,
        "on_time_pmt_streak_12m": on_time_pmt_streak_12m,
        "revolver_ratio_12m": revolver_ratio_12m,
        "spend_mean_6m": spend_mean_6m,
        "spend_vol_6m": spend_vol_6m,
        "score_current": score_current,
        "score_delta_6m": score_delta_6m,
        "inq_6m": inq_6m,
        "dpd_30_12m": dpd_30_12m,
        "dpd_60_12m": dpd_60_12m,
        "proposed_cli": proposed_cli,
        "expected_util_uplift": expected_util_uplift,
        "lgd": lgd,
        "ead": ead,
        "default_6m": default_6m
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_cli_data(n=200_000, as_of_date="2025-12-31")
    df.to_csv("cli_modeling_table.csv", index=False)
    print("Wrote: cli_modeling_table.csv", df.shape)
