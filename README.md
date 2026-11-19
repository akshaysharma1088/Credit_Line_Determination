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

## Headline results (reference seed, test fold)

| | Logistic | XGBoost + isotonic |
|---|---|---|
| AUC | 0.758 | 0.775 |
| KS | 0.404 | 0.432 |
| Net profit at matched 55.5% approval | 387,310 | 421,120 |

| Fairness (single threshold to group thresholds, same loss budget) | Before | After |
|---|---|---|
| Demographic-parity difference | 0.092 | 0.012 |
| Disparate-impact ratio | 0.84 | 1.02 |
| Realized net profit | 421,120 | 414,370 |

Isotonic calibration reduces expected calibration error from 0.270 to 0.008.

**Correction (revision).** In the first release the boosted model was trained with class weights while early stopping monitored unweighted validation log-loss. That rule never triggered, so the model ran all 600 rounds and was over-trained (test AUC 0.762). Early stopping now monitors validation AUC (`src/cli/model.py`), and all results were regenerated.

## Robustness analyses (added in revision)

```bash
bash scripts/get_card_data.sh        # UCI Default of Credit Card Clients (30,000 real card accounts)
python run_all.py --robustness       # about one hour on 2 cores
```

`src/cli/robustness.py`, `src/cli/card.py` and `src/cli/alternatives.py` write `results/10_` to `results/17_` and draw Figs. 6 and 7:

| File | Content |
|---|---|
| `10_multi_seed.json` | Whole pipeline over 30 seeds, including the fixed-budget fairness frontier and the access-expanding policy |
| `11_bootstrap_reference_seed.json` | 95% bootstrap intervals (2,000 resamples) on the reference seed |
| `12_generator_sensitivity.json` | Generator assumptions varied one at a time, plus a 240,000-account run |
| `13_economic_sensitivity.json` | Margin, LGD and budget share varied over the 30 seeds |
| `14_dynamics_robustness.json` | Multi-period simulation over 30 seeds with parameter sensitivity |
| `15_external_repeated_splits.json` | 30 repeated stratified splits of `credit_data` |
| `16_card_data.json` | Full single-period pipeline on real credit-card accounts, 30 splits, sex and age as protected attributes, plus group-blind alternatives |
| `17_group_blind_synthetic.json` | Group-blind alternatives on the synthetic portfolio: feature removal, logistic model, and the fairness-regularized model |

Over 30 seeds: AUC difference (XGBoost minus logistic) 0.016 (95% range 0.012 to 0.019); parity gap 0.113 to 0.009 under the same expected-loss budget at a mean profit cost of 1.2% (-3.5% to +1.5%). Closing the gap by expanding access raises profit by 4.7% but uses 13.3% more expected loss. On the real card accounts XGBoost leads logistic regression by 0.025 AUC, and the parity gap falls from 0.047 to 0.017 (sex) at a profit change of -0.4%. Group-blind alternatives: feature removal closes little of the gap; a fairness-regularized model with one threshold (`src/cli/alternatives.py`) closes most of it on the card accounts and costs 9.2% of profit to do so on the synthetic portfolio.

## Data

- `data/synthetic_portfolio.csv` is generated, not shipped. All records are fictional.
- The card accounts come from the UCI Machine Learning Repository (Yeh 2009, doi 10.24432/C55S3H). Sex, age and marital status are never model inputs.
- `credit_data` comes from the Rdatasets archive (`modeldata` package): https://github.com/vincentarelbundock/Rdatasets

## License

MIT
