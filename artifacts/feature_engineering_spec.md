Feature Engineering Specification
Proactive Credit Line Increase (CLI) Risk Model
1. Overview

This document defines the authoritative feature construction logic for the CLI risk model.

It specifies:

Source tables

Event-time windowing

Aggregation logic

Leakage prevention rules

SQL/PySpark pseudocode

Feature versioning practices

All features must satisfy:

𝑓
𝑒
𝑎
𝑡
𝑢
𝑟
𝑒
_
𝑡
𝑖
𝑚
𝑒
𝑠
𝑡
𝑎
𝑚
𝑝
≤
𝑎
𝑠
_
𝑜
𝑓
_
𝑑
𝑎
𝑡
𝑒
feature_timestamp≤as_of_date

No forward information is permitted.

2. Source Tables
2.1 transactions
Column	Description
acct_id	Account identifier
txn_date	Transaction date
txn_amount	Transaction amount (positive for spend)
mcc	Merchant category code
channel	Transaction channel
2.2 statements
Column	Description
acct_id	Account identifier
statement_date	Statement closing date
statement_balance	Statement balance
payment_amount	Payment made during cycle
days_past_due	DPD at statement date
curr_limit	Credit limit at statement date
2.3 bureau_snapshots
Column	Description
acct_id	Account identifier
snapshot_date	Bureau refresh date
score	Credit score
inquiry_count_6m	External inquiries past 6m
3. Window Definitions

Let:

as_of_date = decision date

Windows defined relative to as_of_date

Window	Definition
3M	(as_of_date - 3 months, as_of_date]
6M	(as_of_date - 6 months, as_of_date]
12M	(as_of_date - 12 months, as_of_date]

All aggregations are computed using statement cycles fully closed before as_of_date.

4. Feature Specifications
4.1 Utilization Features
Definition
𝑢
𝑡
𝑖
𝑙
=
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
util=
curr_limit
statement_balance
	​

SQL (Generic ANSI Style)
WITH util_base AS (
    SELECT
        acct_id,
        statement_date,
        statement_balance / NULLIF(curr_limit, 0) AS util
    FROM statements
    WHERE statement_date <= :as_of_date
)

SELECT
    acct_id,
    AVG(CASE WHEN statement_date > DATEADD(month, -3, :as_of_date) THEN util END) AS util_3m_mean,
    AVG(CASE WHEN statement_date > DATEADD(month, -6, :as_of_date) THEN util END) AS util_6m_mean,
    AVG(CASE WHEN statement_date > DATEADD(month, -12, :as_of_date) THEN util END) AS util_12m_mean,
    STDDEV(CASE WHEN statement_date > DATEADD(month, -12, :as_of_date) THEN util END) AS util_vol_12m
FROM util_base
GROUP BY acct_id;
Utilization Slope (6M)

Preferred approach: linear regression slope over 6 months.

SQL Approximation (difference approach)
SELECT
    acct_id,
    (AVG(CASE WHEN statement_date > DATEADD(month, -3, :as_of_date) THEN util END)
     -
     AVG(CASE WHEN statement_date > DATEADD(month, -12, :as_of_date) THEN util END)
    ) / 6.0 AS util_6m_slope
FROM util_base
GROUP BY acct_id;
PySpark Implementation
from pyspark.sql import functions as F

util_df = (
    statements
    .filter(F.col("statement_date") <= F.lit(as_of_date))
    .withColumn("util", F.col("statement_balance") / F.col("curr_limit"))
)

features = (
    util_df
    .groupBy("acct_id")
    .agg(
        F.avg(F.when(F.col("statement_date") > F.add_months(F.lit(as_of_date), -3), F.col("util"))).alias("util_3m_mean"),
        F.avg(F.when(F.col("statement_date") > F.add_months(F.lit(as_of_date), -6), F.col("util"))).alias("util_6m_mean"),
        F.avg(F.when(F.col("statement_date") > F.add_months(F.lit(as_of_date), -12), F.col("util"))).alias("util_12m_mean"),
        F.stddev(F.when(F.col("statement_date") > F.add_months(F.lit(as_of_date), -12), F.col("util"))).alias("util_vol_12m")
    )
)
4.2 Payment Features
Payment-to-Balance Ratio
𝑃
𝑇
𝑅
=
𝑝
𝑎
𝑦
𝑚
𝑒
𝑛
𝑡
_
𝑎
𝑚
𝑜
𝑢
𝑛
𝑡
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
+
𝜖
PTR=
statement_balance+ϵ
payment_amount
	​

SQL
SELECT
    acct_id,
    AVG(CASE WHEN statement_date > DATEADD(month, -6, :as_of_date)
         THEN payment_amount / NULLIF(statement_balance,0)
    END) AS pmt_to_bal_6m,
    COUNT(CASE WHEN days_past_due = 0
         AND statement_date > DATEADD(month, -12, :as_of_date)
    THEN 1 END) AS on_time_pmt_streak_12m
FROM statements
WHERE statement_date <= :as_of_date
GROUP BY acct_id;
4.3 Revolver Ratio

Definition: proportion of months where balance carried forward.

SELECT
    acct_id,
    SUM(CASE WHEN statement_balance > payment_amount THEN 1 ELSE 0 END)
    /
    COUNT(*) AS revolver_ratio_12m
FROM statements
WHERE statement_date > DATEADD(month, -12, :as_of_date)
GROUP BY acct_id;
4.4 Spend Features

Spend computed from transactions table.

SELECT
    acct_id,
    AVG(CASE WHEN txn_date > DATEADD(month, -6, :as_of_date)
        THEN txn_amount END) AS spend_mean_6m,
    STDDEV(CASE WHEN txn_date > DATEADD(month, -6, :as_of_date)
        THEN txn_amount END) AS spend_vol_6m
FROM transactions
WHERE txn_date <= :as_of_date
GROUP BY acct_id;
4.5 Bureau Features

Latest bureau snapshot prior to as_of_date.

SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY acct_id ORDER BY snapshot_date DESC) AS rn
    FROM bureau_snapshots
    WHERE snapshot_date <= :as_of_date
) t
WHERE rn = 1;

Score delta:

score_current - score_6m_ago AS score_delta_6m
4.6 Delinquency Counts
SELECT
    acct_id,
    COUNT(CASE WHEN days_past_due >= 30 THEN 1 END) AS dpd_30_12m,
    COUNT(CASE WHEN days_past_due >= 60 THEN 1 END) AS dpd_60_12m
FROM statements
WHERE statement_date > DATEADD(month, -12, :as_of_date)
GROUP BY acct_id;
5. Target Label Construction
Forward-Looking Default
SELECT
    acct_id,
    CASE
        WHEN MAX(days_past_due) >= 60 THEN 1
        ELSE 0
    END AS default_6m
FROM statements
WHERE statement_date > :as_of_date
  AND statement_date <= DATEADD(month, 6, :as_of_date)
GROUP BY acct_id;

Important: This query must never be joined back into training data prior to label cutoff.

6. Leakage Prevention Rules

No transaction, statement, or bureau event beyond as_of_date.

No post-CLI behavior.

Use statement_date, not ingestion timestamp.

Freeze curr_limit at decision time.

7. Feature Versioning

Every model training run must record:

Feature version ID

SQL/PySpark commit hash

Data cutoff date

Hyperparameter configuration

Recommended: store in MLflow or metadata registry.

8. Reproducibility Checklist

To regenerate full modeling dataset:

Select as_of_date.

Compute features using window rules.

Compute forward 6M default label.

Join all features into final table.

Save snapshot with version tag.
