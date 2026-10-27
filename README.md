# Credit Line Determination

Reproducibility package for:

> **Responsible Credit-Line Increase: A Fairness-Aware, Explainable, and Governed Machine-Learning Framework for Socio-Technical Lending Systems**
> Akshay Sharma, Gaurav Sharma

One command regenerates every number, table and figure in the paper from a fixed seed. No proprietary data is used: the main study runs on a synthetic credit-card portfolio produced by the included generator, and the external validation uses the public `credit_data` benchmark.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/get_credit_data.sh      # downloads the public benchmark used in Section 7.6
python run_all.py                    # about 30 seconds on a laptop
```

Outputs go to `results/` (JSON, one file per analysis) and `figures/` (PNG). Running twice produces identical results.

## What each module does

| Module | Paper section | Output |
|---|---|---|
| `src/cli/generate.py` | 4.1 Synthetic portfolio | `data/synthetic_portfolio.csv`, `results/01_data_summary.json` |
| `src/cli/model.py` | 5, 6, 7.1 to 7.5 | Discrimination and calibration (Tables 1 and 2, Figs. 2 and 3), SHAP importance (Fig. 4), family ablation (Table 3), economics at matched approval (Table 4, Fig. 5), per-group performance and fairness before and after recalibration (Tables 5 and 6), fairness and economics frontier (Fig. 6) |
| `src/cli/external.py` | 7.6 External validation | Table 7 |
| `src/cli/dynamics.py` | 7.7 Multi-period dynamics | Fig. 7 |
| `src/cli/common.py` | 3 Problem formulation | Seed, economic parameters, metrics |

## Key settings

- Seed: `20261027`
- Portfolio: 60,000 accounts over four years. Out-of-time split: train on years 1 and 2, validate on year 3, test on year 4.
- Models: standardized logistic regression; XGBoost (depth 5, learning rate 0.05, up to 600 rounds with early stopping, subsampling, positive-class weighting) with isotonic calibration fitted on the validation year.
- Economics: standardized increment 1,000, incremental utilization 0.5, margin 0.18, LGD 0.80 (break-even PD 0.225). Expected-loss budget = 60% of the loss of approving every account below break-even.
- Fairness: binary protected-attribute proxy `A` in the synthetic data; age under 30 in `credit_data`.

## Headline results (test fold)

| | Logistic | XGBoost + isotonic |
|---|---|---|
| AUC | 0.758 | 0.762 |
| KS | 0.404 | 0.419 |
| Net profit at matched 51.2% approval | 370,290 | 366,370 |

| Fairness (single threshold to group-recalibrated, same loss budget) | Before | After |
|---|---|---|
| Demographic-parity difference | 0.084 | 0.013 |
| Disparate-impact ratio | 0.85 | 1.03 |
| Realized net profit | 366,370 | 364,210 |

Isotonic calibration reduces expected calibration error from 0.194 to 0.006. In the multi-period simulation the fairness-aware policy narrows the credit-access gap by 43% at a 0.6% cost in cumulative profit.

## Data

- `data/synthetic_portfolio.csv` is generated, not shipped. All records are fictional.
- `credit_data` comes from the Rdatasets archive (`modeldata` package): https://github.com/vincentarelbundock/Rdatasets

## License

MIT
