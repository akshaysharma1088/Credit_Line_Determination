Credit Line Prediction (Proactive CLI) — Data Dictionary
1. Overview

This project models proactive credit line increase (CLI) eligibility by predicting default risk over a forward horizon using behavioral and bureau-driven features computed from historical account activity.

Observation unit: one row per (acct_id, as_of_date) decision snapshot

Feature lookback windows: rolling 3-month, 6-month, and 12-month windows ending at as_of_date

Label horizon: forward 6 months from as_of_date (for default_6m)

Leakage rule: features must use only events with event_date <= as_of_date

2. Table Description
2.1 Primary Modeling Table

File: cli_modeling_table.csv
Primary key: (acct_id, as_of_date)

Each row corresponds to a CLI decision point at as_of_date.

3. Columns and Definitions
3.1 Identifier and Metadata
Column	Type	Description	Notes
acct_id	int/string	Unique account identifier	De-identified
as_of_date	date	Decision snapshot date	All features end at this date
3.2 Account Attributes
Column	Type	Unit	Definition	Windowing / Computation
curr_limit	float	USD	Current credit line at as_of_date	Snapshot field from account master
tenure_months	int	months	Months since account open	floor((as_of_date - open_date)/30)
3.3 Utilization Features

Utilization is defined as:

𝑢
𝑡
𝑖
𝑙
𝑡
=
𝑏
𝑎
𝑙
𝑎
𝑛
𝑐
𝑒
𝑡
𝑐
𝑢
𝑟
𝑟
_
𝑙
𝑖
𝑚
𝑖
𝑡
𝑡
util
t
	​

=
curr_limit
t
	​

balance
t
	​

	​


where balance_t is statement balance or average daily balance (institution choice; must be consistent).

Column	Type	Range	Definition	Windowing / Computation
util_3m_mean	float	0–1.5	Mean utilization over last 3 months	mean(util_m) for months 
𝑚
∈
[
𝑡
−
3
,
𝑡
−
1
]
m∈[t−3,t−1]
util_6m_mean	float	0–1.5	Mean utilization over last 6 months	mean(util_m) for months 
𝑚
∈
[
𝑡
−
6
,
𝑡
−
1
]
m∈[t−6,t−1]
util_12m_mean	float	0–1.5	Mean utilization over last 12 months	mean(util_m) for months 
𝑚
∈
[
𝑡
−
12
,
𝑡
−
1
]
m∈[t−12,t−1]
util_6m_slope	float	-1–1	Trend in utilization over 6 months	Linear regression slope of monthly utilization vs time over last 6 months (preferred); or (util_3m_mean - util_12m_mean)/6 (approx)
util_vol_12m	float	0–0.8	Volatility of utilization over 12 months	std(util_m) for months 
𝑚
∈
[
𝑡
−
12
,
𝑡
−
1
]
m∈[t−12,t−1]

Interpretation notes:

Rising utilization slope often signals increasing financial stress.

High volatility indicates unstable borrowing behavior.

3.4 Payment Behavior Features

Payment ratio is defined as:

𝑃
𝑇
𝑅
𝑚
=
𝑝
𝑎
𝑦
𝑚
𝑒
𝑛
𝑡
𝑚
𝑠
𝑡
𝑎
𝑡
𝑒
𝑚
𝑒
𝑛
𝑡
_
𝑏
𝑎
𝑙
𝑎
𝑛
𝑐
𝑒
𝑚
+
𝜖
PTR
m
	​

=
statement_balance
m
	​

+ϵ
payment
m
	​

	​


(Use a small 
𝜖
ϵ to avoid divide-by-zero.)

Column	Type	Range	Definition	Windowing / Computation
pmt_to_bal_3m	float	0–3	Average payment-to-balance ratio over 3 months	mean(PTR_m) for months 
𝑚
∈
[
𝑡
−
3
,
𝑡
−
1
]
m∈[t−3,t−1]
pmt_to_bal_6m	float	0–3	Average payment-to-balance ratio over 6 months	mean(PTR_m) for months 
𝑚
∈
[
𝑡
−
6
,
𝑡
−
1
]
m∈[t−6,t−1]
on_time_pmt_streak_12m	int	0–12	Count of on-time payments in last 12 cycles	Number of months with dpd_m = 0 over months 
𝑚
∈
[
𝑡
−
12
,
𝑡
−
1
]
m∈[t−12,t−1]

Interpretation notes:

Higher payment-to-balance ratios reduce default risk.

Long on-time streaks strongly indicate stability and suitability for CLI.

3.5 Revolving and Stability Features
Column	Type	Range	Definition	Windowing / Computation
revolver_ratio_12m	float	0–1	Fraction of months where balance was carried forward	count(months with balance_revolve=1)/12
spend_mean_6m	float	USD	Average monthly spend over last 6 months	mean(spend_m) for months 
𝑚
∈
[
𝑡
−
6
,
𝑡
−
1
]
m∈[t−6,t−1]
spend_vol_6m	float	USD	Volatility of monthly spend over last 6 months	std(spend_m) for months 
𝑚
∈
[
𝑡
−
6
,
𝑡
−
1
]
m∈[t−6,t−1]

Interpretation notes:

Revolving behavior can increase profitability but may raise risk if paired with rising utilization and declining payments.

Spend volatility is a proxy for income instability or irregular usage patterns.

3.6 Bureau and Drift Features
Column	Type	Range	Definition	Windowing / Computation
score_current	int	300–850	Credit score at as_of_date	Snapshot (bureau refresh date ≤ as_of_date)
score_delta_6m	int	-150–150	Change in score over last 6 months	score_current - score_6m_ago (last available within window)
inq_6m	int	0–12	Number of inquiries in past 6 months	Count of bureau inquiries within last 6 months

Interpretation notes:

Negative score drift is often a leading indicator of deteriorating credit health.

High inquiries can indicate credit-seeking behavior.

3.7 Delinquency History Features
Column	Type	Range	Definition	Windowing / Computation
dpd_30_12m	int	0–N	Count of 30+ DPD occurrences in last 12 months	Count of monthly cycles where days_past_due >= 30
dpd_60_12m	int	0–N	Count of 60+ DPD occurrences in last 12 months	Count of monthly cycles where days_past_due >= 60

Interpretation notes:

60+ DPD is strongly predictive of future default; this feature is typically one of the most important.

3.8 Decision Simulation / Economic Variables (Optional but Recommended)

These fields support economic evaluation and risk-constrained thresholding.

Column	Type	Unit	Definition	Notes
proposed_cli	float	USD	Proposed credit line increase amount	Used for profit/loss simulation
expected_util_uplift	float	0–1	Expected utilization increase fraction after CLI	Can be estimated via historical uplift or segment models
lgd	float	0–1	Loss given default	Portfolio/segment-level estimate
ead	float	USD	Exposure at default estimate	Often curr_limit * util_6m_mean or balance-based
3.9 Target Label
Column	Type	Values	Definition	Horizon
default_6m	int	0/1	Default event indicator after as_of_date	6 months forward

Recommended default definition (choose one and document):

Charge-off within 6 months, or

60+ DPD within 6 months

4. Windowing Rules and Leakage Prevention
4.1 Feature Time Windows

All behavioral features are computed using only information from:

Lookback window: (as_of_date - W, as_of_date]

Where 
𝑊
∈
{
3
𝑚
,
6
𝑚
,
12
𝑚
}
W∈{3m,6m,12m}

4.2 Label Window

default_6m must use only outcomes in:

(as_of_date, as_of_date + 6 months]

4.3 Leakage Controls

Do not include:

Collections activity after as_of_date

Post-CLI utilization changes (treatment leakage)

Outcome-linked operational flags recorded after the fact

5. Missing Data Handling

Recommended practices:

Numeric features: impute with median or model-native handling (tree models)

Bureau fields: carry-forward last observation (LOCF) if refresh is sparse

Create missingness indicators if missingness is informative (optional)

6. Units and Scaling

Utilization: ratio (0–1.5 allowed for over-limit behavior)

Payment ratio: ratio (0–3 capped)

Spend: USD

Scores: 300–850

Tree models do not require scaling; logistic regression should standardize continuous features.

7. Reproducibility Notes

To reproduce the dataset:

Run 01_generate_synthetic_data.py to generate cli_modeling_table.csv.

Train + evaluate: 02_train_evaluate_visualize.py

Generate explainability figures: 03_shap_figures.py

8. Mapping to Paper Sections

Section 4 (Data & Features): uses Sections 3.2–3.7 above

Section 6 (Explainability): uses SHAP on all non-ID features

Section 7 (Economic evaluation): uses optional fields proposed_cli, expected_util_uplift, lgd, ead
