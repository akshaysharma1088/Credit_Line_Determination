# Credit Line Determination
### Risk-Constrained Explainable Machine Learning for Proactive CLI Decisioning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![XGBoost 2.0](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP 0.44](https://img.shields.io/badge/SHAP-0.44-green.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ESWA Submission](https://img.shields.io/badge/Submitted-Expert%20Systems%20with%20Applications-red.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications)

---

## 📄 Paper

> **Proactive Credit Line Increase Using Risk-Constrained Explainable Machine Learning**
> Akshay Sharma, Gaurav Sharma
> *Submitted to Expert Systems with Applications (Elsevier), May 2026*
> Contact: akshay.sharma1088@gmail.com

This repository provides the complete reproducibility package for the above manuscript, including the synthetic portfolio generator, feature engineering pipeline, model training and calibration scripts, Algorithm 1 implementation, SHAP explainability modules, fairness evaluation, and economic simulation.

---

## 🔍 Overview

Traditional credit line increase (CLI) strategies rely on static scorecards that are blind to short-term behavioral signals. This project formalizes CLI as a **constrained profit maximization problem** under portfolio expected loss and fairness constraints, implemented via a calibrated XGBoost classifier with integrated TreeSHAP explainability — deployed within a governance architecture aligned with SR 11-7 and Basel IRB requirements.

### Key Results (500,000-account synthetic portfolio)

| Metric | Logistic Regression (Baseline) | Proposed (XGBoost + Isotonic) |
|--------|-------------------------------|-------------------------------|
| AUC-ROC | 0.741 ± 0.008 | **0.861 ± 0.006** |
| KS Statistic | 0.312 | **0.443** |
| Brier Score | 0.044 | **0.031** |
| Expected Calibration Error | 0.019 | **0.011** |
| Expected Loss Reduction | — | **−18.3%** |
| Revenue Lift | — | **+11.2%** |
| Net Profit Improvement | — | **+14.7%** |
| Equal Opportunity Difference | 0.071 | **0.028** ✅ (ε=0.05) |

---

## 📁 Repository Structure

```
Credit_Line_Determination/
│
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py      # Gaussian copula portfolio generator
│   │   └── feature_engineering.py      # 47-feature behavioral pipeline
│   │
│   ├── models/
│   │   ├── train.py                    # XGBoost training with time-aware CV
│   │   ├── calibration.py              # Isotonic regression calibration
│   │   └── baselines.py                # Logistic regression + random forest
│   │
│   ├── optimization/
│   │   └── threshold_calibration.py    # Algorithm 1: risk-constrained threshold
│   │
│   ├── explainability/
│   │   └── shap_pipeline.py            # TreeSHAP scoring + adverse action
│   │
│   ├── fairness/
│   │   └── fairness_metrics.py         # EOD, DPD, DIR computation
│   │
│   ├── evaluation/
│   │   ├── discrimination_metrics.py   # AUC, KS, Brier, ECE
│   │   └── economic_simulation.py      # EL, revenue lift, net profit
│   │
│   └── monitoring/
│       └── drift_monitor.py            # PSI-based population drift
│
├── artifacts/
│   ├── feature_spec.yaml               # Feature definitions and window specs
│   ├── model_config.yaml               # Hyperparameter configuration
│   └── threshold_config.yaml           # Risk appetite and fairness parameters
│
├── Paper_Figs/                         # Figures referenced in the manuscript
├── notebooks/
│   └── end_to_end_demo.ipynb           # Full pipeline walkthrough
│
├── docker/
│   └── Dockerfile                      # Reproducibility environment
├── requirements.txt
└── README.md
```

---

## 📊 Dataset: Synthetic Portfolio Design

The evaluation dataset is fully synthetic, generated using a Gaussian copula over credit bureau signals and behavioral features, with conditional default rates calibrated to published revolving credit portfolio benchmarks (Thomas et al., 2017; Siddiqi, 2006). **No real customer data is used or included.**

| Parameter | Specification |
|-----------|--------------|
| Accounts (N) | 500,000 |
| Observation horizon | 48 months (Years 1–4) |
| Label horizon (T) | 12 months forward-looking |
| Default definition | 60+ days past due or charge-off |
| Default rate (class imbalance) | 4.8% |
| Engineered features | 47 across 4 behavioral families |
| Train / Validation / Test | Years 1–2 / Year 3 / Year 4 (time-aware) |
| Generation method | Gaussian copula with calibrated marginals |

Generate the synthetic dataset:

```bash
python src/data/synthetic_generator.py \
  --n_accounts 500000 \
  --horizon_months 48 \
  --default_rate 0.048 \
  --output_path artifacts/synthetic_portfolio.parquet \
  --seed 42
```

---

## 🔧 Feature Engineering: Four Behavioral Families

All features are computed on **event-time windows** (not ingestion timestamps) to prevent training-serving skew. Windows are computed at t−3m, t−6m, and t−12m relative to the decision point.

### 1. Utilization Dynamics
| Feature | Description |
|---------|-------------|
| `util_mean_3m` | Rolling mean utilization — 3-month window |
| `util_mean_6m` | Rolling mean utilization — 6-month window |
| `util_slope_6m` | Utilization trend slope (OLS over 6 months) |
| `util_volatility` | Standard deviation of monthly utilization |

### 2. Payment Stability
| Feature | Description |
|---------|-------------|
| `payment_to_balance_ratio` | PTR = Payment / Statement Balance |
| `consecutive_ontime_payments` | Count of consecutive on-time payments |
| `full_to_min_payment_ratio` | Full payments / Minimum-only payments |
| `payment_timing_deviation_days` | Days deviation from payment due date |

### 3. Behavioral Stability
| Feature | Description |
|---------|-------------|
| `spend_cv` | Coefficient of variation of monthly spend |
| `revolver_consistency` | Stability of revolving behavior indicator |
| `spend_entropy` | Shannon entropy over MCC spend distribution |

### 4. Risk Drift Indicators
| Feature | Description |
|---------|-------------|
| `score_delta_3m` | Credit score change over 3 months |
| `score_delta_6m` | Credit score change over 6 months |
| `inquiry_count_90d` | Bureau inquiries in past 90 days |
| `delinquency_spike_flag` | Short-term delinquency spike indicator |

Full feature specification: [`artifacts/feature_spec.yaml`](artifacts/feature_spec.yaml)

---

## ⚙️ Algorithm 1: Risk-Constrained Threshold Calibration

The core decisioning algorithm selects the optimal PD threshold that simultaneously maximizes portfolio profit while satisfying expected loss and fairness constraints.

```
Algorithm 1: Risk-Constrained Threshold Calibration

Input:  {PD_i, LGD_i, EAD_i, Δ_i} for i=1..N,  EL_target,  fairness tolerance ε
Output: Optimal decision threshold τ*

1. Sort accounts ascending by PD_i.
2. Initialize best_profit ← -inf,  τ* ← null
3. For each candidate τ in {sorted PD values}:
   a.  S(τ) ← { i : PD_i < τ }
   b.  EL(τ) ← Σ_{i∈S(τ)} PD_i · LGD_i · (EAD_i + Δ_i)
   c.  EOD(τ) ← |P(Approve|Y=1,A=0) − P(Approve|Y=1,A=1)|
   d.  If EL(τ) ≤ EL_target AND EOD(τ) ≤ ε:
          Profit(τ) ← Σ_{i∈S(τ)} Π_i
          If Profit(τ) > best_profit:
             best_profit ← Profit(τ);  τ* ← τ
4. Return τ*

Complexity: O(N log N) from initial sort; O(N) for cumulative EL scan.
```

Run threshold calibration:

```bash
python src/optimization/threshold_calibration.py \
  --scores_path artifacts/val_scores.parquet \
  --el_target 16000000 \
  --fairness_tolerance 0.05 \
  --lgd 0.70 \
  --net_margin 0.025
```

---

## 🚀 Quickstart

### Option 1: pip install

```bash
# Clone the repository
git clone https://github.com/akshaysharma1088/Credit_Line_Determination.git
cd Credit_Line_Determination

# Install dependencies
pip install -r requirements.txt

# Generate synthetic portfolio
python src/data/synthetic_generator.py --n_accounts 500000 --seed 42

# Train and calibrate the model
python src/models/train.py --config artifacts/model_config.yaml

# Run threshold calibration (Algorithm 1)
python src/optimization/threshold_calibration.py \
  --el_target 16000000 --fairness_tolerance 0.05

# Generate SHAP explanations
python src/explainability/shap_pipeline.py

# Run full evaluation (Tables 3–6 in the paper)
python src/evaluation/discrimination_metrics.py
python src/evaluation/economic_simulation.py
python src/fairness/fairness_metrics.py
```

### Option 2: Docker (fully reproducible environment)

```bash
docker build -t cli-research ./docker
docker run -v $(pwd)/artifacts:/app/artifacts cli-research python src/models/train.py
```

---

## 📦 Dependencies

```
python==3.10
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.44.1
pyspark==3.5.1
pandas==2.2.1
numpy==1.26.4
scipy==1.13.0
matplotlib==3.8.4
seaborn==0.13.2
pyarrow==15.0.2
```

Full list: [`requirements.txt`](requirements.txt)

---

## 📐 Formal Problem Formulation

CLI is formalized as a constrained profit maximization problem. For account *i* with feature vector **X**_i ∈ ℝᵈ and proposed CLI Δ_i:

**Incremental revenue:**
```
R_i = γ · Δ_i · U_i
```

**Incremental expected loss:**
```
L_i = PD_i · LGD_i · (EAD_i + Δ_i)
```

**Optimization objective:**
```
max_w  Σ_{i∈S} Π_i     s.t.:   EL ≤ EL_target,   EOD ≤ ε,   0 ≤ Δ_i ≤ Δ_i^max
```

Where: γ = 0.025 (net margin rate), LGD = 0.70, mean Δ_i = $2,000, mean U_i = 0.35.

---

## 📏 Reproducibility Checklist

| Component | Status | Script |
|-----------|--------|--------|
| Synthetic data generation | ✅ | `src/data/synthetic_generator.py` |
| Feature engineering pipeline | ✅ | `src/data/feature_engineering.py` |
| Time-aware cross-validation | ✅ | `src/models/train.py` |
| Isotonic calibration | ✅ | `src/models/calibration.py` |
| Algorithm 1 (threshold calibration) | ✅ | `src/optimization/threshold_calibration.py` |
| TreeSHAP explainability | ✅ | `src/explainability/shap_pipeline.py` |
| Fairness metrics (EOD, DPD, DIR) | ✅ | `src/fairness/fairness_metrics.py` |
| Economic simulation | ✅ | `src/evaluation/economic_simulation.py` |
| PSI drift monitoring | ✅ | `src/monitoring/drift_monitor.py` |
| Docker environment | ✅ | `docker/Dockerfile` |

All tables and figures in the manuscript are regeneratable from these scripts using `--seed 42`.

---

## 📊 Reproducing Paper Results

Running the full pipeline reproduces the following results from the manuscript:

**Table 3 — Discrimination Performance**
```bash
python src/evaluation/discrimination_metrics.py --test_year 4
```

**Table 4 — Ablation Study**
```bash
python src/evaluation/discrimination_metrics.py --ablation True
```

**Table 5 — Economic Evaluation**
```bash
python src/evaluation/economic_simulation.py --el_target 16000000
```

**Table 6 — Fairness Results**
```bash
python src/fairness/fairness_metrics.py --tolerance 0.05
```

---

## 🏛️ Governance Architecture

The system is designed for SR 11-7 model risk management compliance:

| SR 11-7 Pillar | Implementation |
|---------------|----------------|
| Conceptual soundness | Mathematical derivation in paper Section 3; feature rationale Section 4.4 |
| Independent validation | Champion/challenger model registry support |
| Ongoing monitoring | PSI drift detection (`src/monitoring/drift_monitor.py`) |
| Change management | Version-controlled features and model artifacts in `artifacts/` |

---

## 📜 Citation

If you use this code or synthetic data generator in your research, please cite:

```bibtex
@article{sharma2026cli,
  title   = {Proactive Credit Line Increase Using Risk-Constrained Explainable Machine Learning},
  author  = {Sharma, Akshay and Sharma, Gaurav},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Submitted for review},
  url     = {https://github.com/akshaysharma1088/Credit_Line_Determination}
}
```

---

## 📬 Contact

**Akshay Sharma** — Principal Data Engineer | IEEE Senior Member | IET Member
- Email: akshay.sharma1088@gmail.com
- LinkedIn: [linkedin.com/in/akshay-sharma-1088](https://linkedin.com/in/akshay-sharma-1088)
- Medium: [medium.com/@akshay.sharma1088](https://medium.com/@akshay.sharma1088)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> **Data Privacy Notice:** This repository contains only synthetic data and code. No real customer records, account numbers, or personally identifiable information are included or referenced.
