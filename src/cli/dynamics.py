"""Stylized multi-period feedback simulation (Section 7.7).

A closed cohort of 20,000 accounts over eight periods. Approved increases raise
limits and incremental exposure; defaults realize and remove accounts; stress
drifts in response to access (moderate new access to low-stress borrowers
reduces stress, repeated or high-stress extension increases it). Two policies
approve the same number of accounts each period: a single PD threshold, and a
fairness-aware policy that equalizes approval rates across groups.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import SEED, FIGS, GAMMA, LGD, INCREMENT, UTIL_INC, save, r

N, PERIODS, APPROVE_SHARE = 20_000, 8, 0.50


def simulate(policy, seed=SEED):
    rng = np.random.default_rng(seed)               # identical cohort and shocks for both policies
    A = (rng.random(N) < 0.40).astype(int)
    stress = rng.normal(0.35 * A, 1.0, N)
    limit = np.clip(np.exp(8.6 - 0.18 * stress - 0.20 * A + rng.normal(0, 0.45, N)), 500, 50_000)
    exposure = np.zeros(N); n_inc = np.zeros(N); alive = np.ones(N, bool)
    shocks = rng.normal(0, 0.10, (PERIODS, N)); model_noise = rng.normal(0, 0.30, (PERIODS, N))
    draws = rng.random((PERIODS, N))
    hist = {"access_gap": [], "approval_gap": [], "cum_profit": []}; cum = 0.0
    for t in range(PERIODS):
        pd_true = 1 / (1 + np.exp(-(-2.9 + 0.9 * stress)))
        score = pd_true * np.exp(model_noise[t])
        k = int(APPROVE_SHARE * alive.sum())
        approve = np.zeros(N, bool)
        if policy == "single":
            cand = np.where(alive)[0]; approve[cand[np.argsort(score[cand])[:k]]] = True
        else:
            for g in (0, 1):
                cand = np.where(alive & (A == g))[0]
                kg = int(round(k * len(cand) / alive.sum()))
                approve[cand[np.argsort(score[cand])[:kg]]] = True
        appr_rate = [approve[alive & (A == g)].mean() for g in (0, 1)]
        limit[approve] += INCREMENT; exposure[approve] += INCREMENT * UTIL_INC; n_inc[approve] += 1
        default = alive & (draws[t] < pd_true)
        cum += (GAMMA / 2) * exposure[alive & ~default].sum() - LGD * exposure[default].sum()
        alive &= ~default
        helped = approve & (stress < 0.5) & (n_inc <= 2)
        strained = approve & ~helped
        stress = stress - 0.06 * helped + 0.08 * strained + shocks[t]
        hist["access_gap"].append(float(limit[alive & (A == 0)].mean() - limit[alive & (A == 1)].mean()))
        hist["approval_gap"].append(float(appr_rate[0] - appr_rate[1]))
        hist["cum_profit"].append(float(cum))
    return hist


def main():
    single, fair = simulate("single"), simulate("fair")
    out = {"single": single, "fair": fair,
           "summary": {"final_access_gap_single": r(single["access_gap"][-1], 0),
                       "final_access_gap_fair": r(fair["access_gap"][-1], 0),
                       "initial_access_gap_single": r(single["access_gap"][0], 0),
                       "approval_gap_single_first_last": [r(single["approval_gap"][0]), r(single["approval_gap"][-1])],
                       "approval_gap_fair_max": r(max(abs(x) for x in fair["approval_gap"])),
                       "cum_profit_single": r(single["cum_profit"][-1], 0), "cum_profit_fair": r(fair["cum_profit"][-1], 0),
                       "profit_change_pct": r(100 * (fair["cum_profit"][-1] / single["cum_profit"][-1] - 1), 1),
                       "access_gap_reduction_pct": r(100 * (1 - fair["access_gap"][-1] / single["access_gap"][-1]), 1)}}
    save("09_dynamics", out)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8)); t = np.arange(1, PERIODS + 1)
    ax[0].plot(t, single["access_gap"], "o-", label="Single threshold"); ax[0].plot(t, fair["access_gap"], "s-", label="Fairness-aware")
    ax[0].set_xlabel("Period"); ax[0].set_ylabel("Mean limit gap (A0 minus A1)"); ax[0].legend(fontsize=7)
    ax[1].plot(t, np.array(single["cum_profit"]) / 1e6, "o-", label="Single threshold")
    ax[1].plot(t, np.array(fair["cum_profit"]) / 1e6, "s-", label="Fairness-aware")
    ax[1].set_xlabel("Period"); ax[1].set_ylabel("Cumulative profit (millions)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig7_dynamics.png", dpi=300); plt.close(fig)
    print(out["summary"])


if __name__ == "__main__":
    main()
