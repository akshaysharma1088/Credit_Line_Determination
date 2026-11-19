"""Robustness analyses added in revision (Section 7.8 and the uncertainty columns of Section 7).

1. Multi-seed replication: the whole pipeline (new portfolio, both models, calibration,
   thresholds, fairness control) is repeated over N_SEEDS independent seeds.
2. Paired bootstrap confidence intervals on the test fold of the reference seed.
3. Fairness-profit frontier under a FIXED expected-loss budget, next to the
   access-expanding frontier of the original submission, with the expected loss of each.
4. Sensitivity of the main findings to the generator's assumptions and to the
   economic parameters.
5. Multi-seed and parameter sensitivity of the multi-period simulation.
6. Repeated stratified splits of the real public benchmark.

Run after cli.generate:  python -m cli.robustness   (or python run_all.py --robustness)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from .common import SEED, GAMMA, LGD, EAD_INC, EL_BUDGET_SHARE, ece, ks_stat, save
from .generate import generate
from .model import fit_lr, fit_xgb, split, fairness
from .common import FEATURES
from . import dynamics, external

N_SEEDS = 30          # multi-seed replication
N_SEEDS_SENS = 10     # seeds per sensitivity configuration
N_BOOT = 2000
GAP_FRACS = [1.0, 0.75, 0.5, 0.25, 0.0]   # share of the single-threshold approval gap that is kept


def fast_auc(y, p):
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)),
            "share_positive": float((x > 0).mean()), "n": int(len(x))}


# ---------------------------------------------------------------- scoring
def fit_scores(df, seed):
    tr, va, te = split(df)
    ytr, yva, yte = tr["default"].values, va["default"].values, te["default"].values
    lr = fit_lr(tr[FEATURES], ytr)
    gb = fit_xgb(tr[FEATURES], ytr, va[FEATURES], yva, seed=seed)
    raw_va, raw_te = gb.predict_proba(va[FEATURES])[:, 1], gb.predict_proba(te[FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, yva)
    S = {"yva": yva, "yte": yte, "Ava": va["A"].values, "Ate": te["A"].values,
         "raw_va": raw_va, "raw_te": raw_te, "cal_va": iso.predict(raw_va), "cal_te": iso.predict(raw_te),
         "lr_te": lr.predict_proba(te[FEATURES])[:, 1]}
    if "pd_true" in te:
        S["oracle_auc"] = fast_auc(yte, te["pd_true"].values)
    return S


# ---------------------------------------------------------------- policies
def _realized(y, a, gamma, lgd):
    rev = gamma * EAD_INC * ((y == 0) & a).sum(); loss = lgd * EAD_INC * ((y == 1) & a).sum()
    return float(rev - loss), float(loss)


def _el(pd_hat, a, lgd):
    return float((pd_hat[a] * lgd * EAD_INC).sum())


def policies(S, gamma=GAMMA, lgd=LGD, share=EL_BUDGET_SHARE):
    """Rebuild the paper's policies on validation and return test-fold approval masks."""
    raw_va, cal_va, Ava = S["raw_va"], S["cal_va"], S["Ava"]
    raw_te, Ate = S["raw_te"], S["Ate"]
    be = gamma / lgd
    budget = share * (cal_va[cal_va < be] * lgd * EAD_INC).sum()
    grid = np.linspace(0.05, 0.95, 901)
    # single threshold (Section 7.4)
    q_star = max(q for q in grid if (cal_va[raw_va <= np.quantile(raw_va, q)] * lgd * EAD_INC).sum() <= budget)
    raw_tau = np.quantile(raw_va, q_star)
    single = raw_te <= raw_tau
    # equal-approval recalibration within the same budget (Section 7.5, Table 6)
    best = None
    for q in grid:
        t0, t1 = np.quantile(raw_va[Ava == 0], q), np.quantile(raw_va[Ava == 1], q)
        appr = np.where(Ava == 1, raw_va <= t1, raw_va <= t0)
        if (cal_va[appr] * lgd * EAD_INC).sum() <= budget:
            best = (t0, t1)
    recal = np.where(Ate == 1, raw_te <= best[1], raw_te <= best[0])
    # access-expanding policy of the original frontier: A0 fixed, A1 raised to the A0 approval rate.
    # This adds approvals, so it spends MORE expected loss than the budget.
    t1x = np.quantile(raw_te[Ate == 1], single[Ate == 0].mean())
    expand = np.where(Ate == 1, raw_te <= t1x, single)
    # fixed-budget frontier: shrink the validation approval gap from its single-threshold value to zero
    # in steps; at each step take the largest approval rates whose expected loss fits the SAME budget.
    sv = raw_va <= raw_tau
    g0 = sv[Ava == 0].mean() - sv[Ava == 1].mean()
    frontier = {}
    for frac in GAP_FRACS:
        g = frac * g0; bestq = None
        for q in grid:
            if not 0.0 < q - g < 1.0:
                continue
            t0, t1 = np.quantile(raw_va[Ava == 0], q), np.quantile(raw_va[Ava == 1], q - g)
            appr = np.where(Ava == 1, raw_va <= t1, raw_va <= t0)
            if (cal_va[appr] * lgd * EAD_INC).sum() <= budget:
                bestq = (t0, t1)
        frontier[frac] = np.where(Ate == 1, raw_te <= bestq[1], raw_te <= bestq[0])
    return {"single": single, "recal": recal, "expand": expand, "frontier": frontier,
            "tau_cal": float(np.interp(raw_tau, np.sort(raw_va), np.sort(cal_va)))}


def evaluate(S, gamma=GAMMA, lgd=LGD, share=EL_BUDGET_SHARE, pol=None):
    y, A, cal = S["yte"], S["Ate"], S["cal_te"]
    pol = pol or policies(S, gamma, lgd, share)
    single, recal, expand = pol["single"], pol["recal"], pol["expand"]
    rate = single.mean()
    lr_appr = S["lr_te"] <= np.quantile(S["lr_te"], rate)
    p_s, l_s = _realized(y, single, gamma, lgd); p_lr, _ = _realized(y, lr_appr, gamma, lgd)
    p_r, l_r = _realized(y, recal, gamma, lgd); p_x, l_x = _realized(y, expand, gamma, lgd)
    fb, fa = fairness(y, single, A), fairness(y, recal, A)
    out = {
        "auc_lr": fast_auc(y, S["lr_te"]), "auc_gb": fast_auc(y, cal),
        "ks_lr": ks_stat(y, S["lr_te"]), "ks_gb": ks_stat(y, cal),
        "ece_raw": ece(y, S["raw_te"]), "ece_cal": ece(y, cal),
        "approval_rate": float(rate),
        "profit_gb_vs_lr_pct": 100 * (p_s / p_lr - 1),
        "dp_before": fb["dp_diff"], "dp_after": fa["dp_diff"], "di_before": fb["di_ratio"], "di_after": fa["di_ratio"],
        "tpr_gap_before": fb["tpr_gap"], "tpr_gap_after": fa["tpr_gap"],
        "fpr_gap_before": fb["fpr_gap"], "fpr_gap_after": fa["fpr_gap"],
        "auc_A0": fast_auc(y[A == 0], cal[A == 0]), "auc_A1": fast_auc(y[A == 1], cal[A == 1]),
        # equal-approval recalibration, same expected-loss budget
        "recal_profit_pct": 100 * (p_r / p_s - 1),
        "recal_expected_loss_pct": 100 * (_el(cal, recal, lgd) / _el(cal, single, lgd) - 1),
        "recal_realized_loss_pct": 100 * (l_r / l_s - 1),
        "recal_approval_pct": 100 * (recal.mean() / rate - 1),
        # access-expanding policy (original Fig. 6 end point): larger risk budget
        "expand_profit_pct": 100 * (p_x / p_s - 1),
        "expand_expected_loss_pct": 100 * (_el(cal, expand, lgd) / _el(cal, single, lgd) - 1),
        "expand_realized_loss_pct": 100 * (l_x / l_s - 1),
        "expand_approval_pct": 100 * (expand.mean() / rate - 1),
    }
    for frac, a in pol["frontier"].items():
        tag = f"{frac:.2f}"
        p, _ = _realized(y, a, gamma, lgd)
        out[f"fb_profit_pct_keep_{tag}"] = 100 * (p / p_s - 1)
        out[f"fb_dp_keep_{tag}"] = fairness(y, a, A)["dp_diff"]
        out[f"fb_el_pct_keep_{tag}"] = 100 * (_el(cal, a, lgd) / _el(cal, single, lgd) - 1)
    if "oracle_auc" in S:
        out["oracle_auc"] = S["oracle_auc"]
    out["auc_gb_minus_lr"] = out["auc_gb"] - out["auc_lr"]
    out["ks_gb_minus_lr"] = out["ks_gb"] - out["ks_lr"]
    return out


def table(rows):
    df = pd.DataFrame(rows)
    return {c: summarize(df[c]) for c in df.columns}


# ---------------------------------------------------------------- analyses
def multi_seed():
    rows, econ = [], {}
    grids = {"margin": [("gamma", g) for g in (0.12, 0.15, 0.18, 0.21, 0.24)],
             "lgd": [("lgd", v) for v in (0.60, 0.70, 0.80, 0.90)],
             "budget_share": [("share", v) for v in (0.40, 0.60, 0.80, 1.00)]}
    for i in range(N_SEEDS):
        seed = SEED + i
        S = fit_scores(generate(seed=seed, keep_truth=True), seed)
        rows.append(evaluate(S))
        for name, g in grids.items():           # economic-parameter sensitivity reuses the fitted scores
            for k, v in g:
                econ.setdefault(f"{name}={v}", []).append(evaluate(S, **{k: v}))
        print("seed", i, round(rows[-1]["auc_gb_minus_lr"], 4), round(rows[-1]["recal_profit_pct"], 2), flush=True)
    keep = ["approval_rate", "profit_gb_vs_lr_pct", "dp_before", "dp_after", "recal_profit_pct",
            "expand_profit_pct", "expand_expected_loss_pct"]
    save("10_multi_seed", {"n_seeds": N_SEEDS, "first_seed": SEED, "summary": table(rows), "per_seed": rows})
    save("13_economic_sensitivity", {k: {c: v for c, v in table(r).items() if c in keep} for k, r in econ.items()})


def bootstrap():
    df = pd.read_csv(__import__("cli.common", fromlist=["DATA"]).DATA / "synthetic_portfolio.csv")
    S = fit_scores(df, SEED); pol = policies(S)
    point = evaluate(S, pol=pol)
    rng = np.random.default_rng(SEED); n = len(S["yte"]); rows = []
    keys = ["yte", "Ate", "raw_te", "cal_te", "lr_te"]
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        Sb = {**S, **{k: S[k][ix] for k in keys}}
        pb = {"single": pol["single"][ix], "recal": pol["recal"][ix], "expand": pol["expand"][ix],
              "frontier": {e: a[ix] for e, a in pol["frontier"].items()}}
        rows.append(evaluate(Sb, pol=pb))
    bt = pd.DataFrame(rows)
    out = {k: {"point": float(point[k]), "lo": float(np.percentile(bt[k], 2.5)), "hi": float(np.percentile(bt[k], 97.5))}
           for k in bt.columns}
    save("11_bootstrap_reference_seed", {"n_boot": N_BOOT, "seed": SEED, "tau_cal": pol["tau_cal"], "ci95": out})


def generator_sensitivity():
    grid = {"w_unobs": [0.30, 0.60, 0.90, 1.20], "nl_scale": [0.0, 0.5, 1.0, 1.5, 2.0],
            "group_shift": [0.0, 0.175, 0.35, 0.70], "proxy_scale": [0.0, 1.0, 2.0]}
    keep = ["oracle_auc", "auc_lr", "auc_gb", "auc_gb_minus_lr", "profit_gb_vs_lr_pct", "ece_cal", "approval_rate",
            "dp_before", "di_before", "dp_after", "recal_profit_pct", "expand_profit_pct", "expand_expected_loss_pct",
            "fb_profit_pct_keep_0.50", "fb_dp_keep_0.50"]
    out = {}
    for knob, vals in grid.items():
        for v in vals:
            rows = []
            for i in range(N_SEEDS_SENS):
                seed = SEED + i
                rows.append(evaluate(fit_scores(generate(seed=seed, keep_truth=True, **{knob: v}), seed)))
            out[f"{knob}={v}"] = {c: s for c, s in table(rows).items() if c in keep}
            print(knob, v, round(out[f"{knob}={v}"]["auc_gb_minus_lr"]["mean"], 4), flush=True)
    # larger training sample: does boosting pull ahead when it has more data?
    rows = []
    for i in range(5):
        seed = SEED + i
        rows.append(evaluate(fit_scores(generate(n=240_000, seed=seed, keep_truth=True), seed)))
    out["n=240000"] = {c: s for c, s in table(rows).items() if c in keep}
    save("12_generator_sensitivity", {"seeds_per_config": N_SEEDS_SENS, "configs": out})


def dynamics_robustness():
    def one(seed, **kw):
        s, f = dynamics.simulate("single", seed, **kw), dynamics.simulate("fair", seed, **kw)
        b = s["baseline_limit_gap"]
        return {"baseline_limit_gap": b, "final_gap_single": s["access_gap"][-1], "final_gap_fair": f["access_gap"][-1],
                "single_vs_baseline_pct": 100 * (s["access_gap"][-1] / b - 1),
                "fair_vs_baseline_pct": 100 * (f["access_gap"][-1] / b - 1),
                "fair_vs_single_pct": 100 * (f["access_gap"][-1] / s["access_gap"][-1] - 1),
                "granted_gap_single": s["granted_gap"][-1], "granted_gap_fair": f["granted_gap"][-1],
                "approval_gap_first": s["approval_gap"][0], "approval_gap_last": s["approval_gap"][-1],
                "profit_pct": 100 * (f["cum_profit"][-1] / s["cum_profit"][-1] - 1),
                "loss_pct": 100 * (f["cum_loss"][-1] / s["cum_loss"][-1] - 1)}
    out = {"base": table([one(SEED + i) for i in range(N_SEEDS)])}
    for name, kws in {"feedback": [0.0, 0.5, 2.0], "noise_sd": [0.15, 0.60], "approve_share": [0.30, 0.70]}.items():
        for v in kws:
            out[f"{name}={v}"] = table([one(SEED + i, **{name: v}) for i in range(N_SEEDS_SENS)])
    save("14_dynamics_robustness", out)


def external_repeats(n=N_SEEDS):
    X, y, age = external.load(); rows = []
    for i in range(n):
        seed = SEED + i; idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
        lr = fit_lr(X.iloc[tr], y[tr]); gb = fit_xgb(X.iloc[tr], y[tr], X.iloc[va], y[va], seed=seed)
        p_lr = lr.predict_proba(X.iloc[te])[:, 1]
        raw_va, raw_te = gb.predict_proba(X.iloc[va])[:, 1], gb.predict_proba(X.iloc[te])[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip").fit(raw_va, y[va]).predict(raw_te)
        rate = 1 - y[va].mean(); A_te = (age[te] < 30).astype(int); A_va = (age[va] < 30).astype(int)
        single = raw_te <= np.quantile(raw_va, rate)
        recal = np.where(A_te == 1, raw_te <= np.quantile(raw_va[A_va == 1], rate), raw_te <= np.quantile(raw_va[A_va == 0], rate))
        fb, fa = fairness(y[te], single, A_te), fairness(y[te], recal, A_te)
        rows.append({"auc_lr": fast_auc(y[te], p_lr), "auc_gb": fast_auc(y[te], raw_te),
                     "auc_gb_minus_lr": fast_auc(y[te], raw_te) - fast_auc(y[te], p_lr),
                     "ks_lr": ks_stat(y[te], p_lr), "ks_gb": ks_stat(y[te], raw_te),
                     "ece_raw": ece(y[te], raw_te), "ece_cal": ece(y[te], cal),
                     "dp_before": fb["dp_diff"], "dp_after": fa["dp_diff"], "di_before": fb["di_ratio"], "di_after": fa["di_ratio"],
                     "fpr_gap_before": fb["fpr_gap"], "fpr_gap_after": fa["fpr_gap"],
                     "tpr_gap_before": fb["tpr_gap"], "tpr_gap_after": fa["tpr_gap"]})
    save("15_external_repeated_splits", {"n_splits": n, "summary": table(rows)})


def group_blind(n=N_SEEDS_SENS):
    """Group-blind less discriminatory alternatives on the synthetic portfolio (Section 7.9)."""
    from . import alternatives
    from .common import FAMILIES
    cands = {f"drop {c}": [c] for c in FEATURES}
    cands.update({f"drop family {k}": v for k, v in FAMILIES.items()})
    cands["drop limit and score"] = ["credit_limit", "bureau_score"]
    cands["logistic model"] = "__lr__"
    rows, picks, grp, reg = [], {}, [], []
    for i in range(n):
        seed = SEED + i
        df = generate(seed=seed); tr, va, te = split(df)
        a = alternatives.search(tr[FEATURES], tr["default"].values, va[FEATURES], va["default"].values,
                                te[FEATURES], te["default"].values, va["A"].values, te["A"].values, cands, seed)
        picks[a["selected"]] = picks.get(a["selected"], 0) + 1
        rows.append({k: a[k] for k in ("gap_full", "gap_selected", "gap_reduction_pct", "profit_pct", "realized_loss_pct")})
        grp.append(a["candidates"])
        reg.append(alternatives.regularized(tr[FEATURES], tr["default"].values, tr["A"].values, va[FEATURES], va["default"].values,
                                            va["A"].values, te[FEATURES], te["default"].values, te["A"].values, seed))
        print("group-blind seed", i, a["selected"], round(a["gap_selected"], 3), flush=True)
    cand_mean = {c: {"gap": float(np.mean([g[c]["gap"] for g in grp])), "profit_pct": float(np.mean([g[c]["profit_pct"] for g in grp]))}
                 for c in grp[0]}
    save("17_group_blind_synthetic", {"n_seeds": n, "tolerance": alternatives.TOL, "summary": table(rows),
                                      "selected_counts": picks, "candidate_means": cand_mean, "regularized": table(reg)})


def main(which=None):
    steps = {"boot": bootstrap, "dyn": dynamics_robustness, "ext": external_repeats,
             "seeds": multi_seed, "gen": generator_sensitivity, "blind": group_blind, "figs": figures}
    for k, f in steps.items():
        if which is None or k in which:
            f(); print("finished", k, flush=True)



def figures():
    """Fig. 6 (frontier under a fixed budget vs access expansion) and Fig. 7 (dynamics with seed bands)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .common import RESULTS, FIGS
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    rows = pd.DataFrame(json.load(open(RESULTS / "10_multi_seed.json"))["per_seed"])
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.9))
    tags = [f"{f:.2f}" for f in GAP_FRACS]
    x = [rows[f"fb_dp_keep_{t}"].mean() for t in tags]
    y = [rows[f"fb_profit_pct_keep_{t}"].mean() for t in tags]
    lo = [np.percentile(rows[f"fb_profit_pct_keep_{t}"], 2.5) for t in tags]
    hi = [np.percentile(rows[f"fb_profit_pct_keep_{t}"], 97.5) for t in tags]
    ax[0].fill_between(x, lo, hi, color="#2F6DB5", alpha=0.18, lw=0, label="95% range over 30 seeds")
    ax[0].plot(x, y, "o-", ms=3.5, color="#2F6DB5", label="Mean over 30 seeds")
    ax[0].axhline(0, color="k", lw=0.7, ls="--"); ax[0].invert_xaxis()
    ax[0].set_xlabel("Demographic-parity gap (test fold)"); ax[0].set_ylabel("Net profit vs single threshold (%)")
    ax[0].set_title("(a) Same expected-loss budget"); ax[0].legend(fontsize=7, loc="lower left")
    lab = ["Parity,\nsame budget", "Parity by\naccess expansion"]
    pr = [rows["recal_profit_pct"], rows["expand_profit_pct"]]
    el = [rows["recal_expected_loss_pct"], rows["expand_expected_loss_pct"]]
    w = 0.36; xs = np.arange(2)
    for k, (vals, name, col) in enumerate([(pr, "Net profit", "#2F6DB5"), (el, "Expected loss", "#C8553D")]):
        m = [v.mean() for v in vals]
        err = [[mm - np.percentile(v, 2.5) for mm, v in zip(m, vals)], [np.percentile(v, 97.5) - mm for mm, v in zip(m, vals)]]
        ax[1].bar(xs + (k - 0.5) * w, m, w, yerr=err, capsize=2.5, color=col, label=name, error_kw={"lw": 0.8})
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_xticks(xs); ax[1].set_xticklabels(lab)
    ax[1].set_ylabel("Change vs single threshold (%)"); ax[1].set_title("(b) Profit and risk budget used"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig6_frontier.png", dpi=300); plt.close(fig)

    runs = {p: [dynamics.simulate(p, SEED + i) for i in range(N_SEEDS)] for p in ("single", "fair")}
    t = np.arange(0, dynamics.PERIODS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    for p, name, col, mk in [("single", "Single threshold", "#C8553D", "o"), ("fair", "Fairness-aware", "#2F6DB5", "s")]:
        g = np.array([[r["baseline_limit_gap"]] + r["access_gap"] for r in runs[p]])
        ax[0].fill_between(t, np.percentile(g, 2.5, 0), np.percentile(g, 97.5, 0), color=col, alpha=0.18, lw=0)
        ax[0].plot(t, g.mean(0), marker=mk, ms=3, color=col, label=name)
        c = np.array([r["cum_profit"] for r in runs[p]]) / 1e6
        ax[1].fill_between(t[1:], np.percentile(c, 2.5, 0), np.percentile(c, 97.5, 0), color=col, alpha=0.18, lw=0)
        ax[1].plot(t[1:], c.mean(0), marker=mk, ms=3, color=col, label=name)
    ax[0].axhline(np.mean([r["baseline_limit_gap"] for r in runs["single"]]), color="k", lw=0.7, ls="--")
    ax[0].set_xlabel("Period (0 = before any decision)"); ax[0].set_ylabel("Mean limit gap (A0 minus A1)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("Period"); ax[1].set_ylabel("Cumulative profit (millions)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "fig7_dynamics.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
