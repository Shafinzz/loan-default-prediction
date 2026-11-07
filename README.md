# Loan Default Prediction at Scale

**Databricks · PySpark · Spark ML Pipelines · MLflow Tracking · Unity Catalog Model Registry**

An end-to-end machine learning project that predicts which personal loans will default, built entirely on Databricks.
It replaces a blunt one-rule credit policy with a model that uses the full application, and proves the gain with numbers.

> Personal project, based on a Databricks ML challenge brief. All analysis, code and decisions are my own.

---

## Results at a glance

| | Value |
|---|---|
| Loans analysed | 1.48M (658K with a final outcome) |
| Best model | Gradient-Boosted Trees in a Spark ML Pipeline |
| Test AUC-ROC (2016 loans, never seen in training) | **0.706** |
| Defaults caught at a 29% reject rate | **50%** (precision 38% vs 22% at random) |
| vs. simple "reject grades D–G" rule, same reject rate | **~490 more defaulters caught per year** (~$8M of risky loan volume) |
| MLflow runs | 9 runs, 3 model types (Logistic Regression, Random Forest, GBT) |
| Registered model | `workspace.loan_default.loan_default_gbt` (`@champion`) |

---

## The business problem

A peer-to-peer lender loses money on roughly 1 in 5 completed loans. Its current policy is a single blunt rule
(reject applicants below a credit-score cut-off), which rejects good borrowers and still misses many defaulters.
The goal: a model that ranks applicants by default risk, beats the simple rule with evidence, and is ready to be deployed.

---

## Approach

```
Lending Club data (Databricks sample)
      │
01 EDA ───────── data-source check · leakage hunt (29 columns) · class imbalance · data quality report
      │
02 Features ──── Spark ML Pipeline: SQLTransformer (clean) → Imputer → StringIndexer → OneHotEncoder → VectorAssembler
      │          time-based split: train 2012–2015 · test 2016
03 Training ──── class weights for imbalance · time-based validation (2015) · 9 MLflow runs · GBT selected
      │
04 Evaluation ── AUC-ROC / precision / recall / F1 · confusion matrix in $ · PR curve & cut-off choice
      │          fair baseline comparison · feature importance
05 Registry ──── Unity Catalog Model Registry · @champion alias · batch scoring · model card
```

---

## Key decisions (and why)

| Decision | Reason |
|---|---|
| **Rejected the suggested 2018 Q2 file** for a 2012–2017 dataset | 93.65% of its loans were still "Current" (outcome unknown), leaving only 134 defaults to learn from |
| **Dropped 29 leakage columns** (payments, recoveries, hardship…) | They are only known *after* the loan starts, so they would make the model look great in testing and fail in production |
| **Removed unfinished loans** (Current, Late, Grace Period) from the target | Their outcome is unknown; labelling them either way would teach the model something false |
| **Time-based split** (train 2012–2015, test 2016; 2017 excluded) | Mimics real use (learn from the past, predict the future). 2017 was excluded because its few finished loans are mostly early payoffs (3.6% default rate) |
| **Class weights** (defaults ×3.82) | Handles the 79/21 imbalance without duplicating or discarding real loans |
| **Baseline = grade rule** instead of "FICO < 650" | The dataset has no FICO column; grade is Lending Club's own risk rating and an equally simple one-column rule |
| **Location columns excluded** | To avoid treating applicants differently based on where they live |

---

## Key findings

- **Default rate rises steadily with grade:** ~6.5% (A) → ~50% (G).
- **60-month loans default about twice as often** as 36-month loans (~33.5% vs ~16.5%).
- **Top drivers:** interest rate (Lending Club's risk pricing), debt-to-income, income, loan amount, mortgage accounts, term,
  and an engineered flag for **unknown employment length**, which ranked in the top 10.
- **A more complex model is not automatically better:** Logistic Regression (AUC 0.703) came very close to GBT (0.706).
- **Hidden data traps caught:** missing values disguised as `"n/a"`, impossible values (`dti = 999`), numbers stored as text
  with hidden spaces, and summary rows inside the raw file.

---

## Repository structure

| Notebook | Purpose |
|---|---|
| `00_config` | All paths, table names, feature lists and the MLflow experiment, in one place |
| `01_eda` | Part A: data-source selection. Part B: leakage, target, insights, data quality report |
| `02_feature_engineering` | Target, cleaning rules, the 5-stage Spark ML Pipeline, time-based split |
| `03_ml_training` | Imbalance strategy, validation design, 9 MLflow runs, final models |
| `04_evaluation` | Test metrics, confusion matrix, PR curve, baseline comparison, feature importance |
| `05_model_registry` | Model registration, description, `@champion` alias, batch predictions, **model card** |

---

## How to run

1. Clone this repo into a Databricks **Git folder** (works on Databricks Free Edition, serverless compute).
2. Run the notebooks **in order**, `00` → `05`. Each one loads shared settings with `%run ./00_config`.
3. Tables are created in `workspace.loan_default`; MLflow runs go to `/Users/<you>/loan_default_experiments`.

---

## Limitations

No credit score (FICO) in the data; strong reliance on Lending Club's own interest-rate pricing; historical data
(2012–2016) that may have drifted; probabilities not calibrated (class weights inflate scores); fairness not yet audited.
See the **model card** in `05_model_registry` for details and production-readiness steps.

---

## Data and credit

Lending Club loan data from [Kaggle (Wendy Kan)](https://www.kaggle.com/wendykan/lending-club-loan-data),
in the Parquet version provided by Databricks as a built-in sample (`/databricks-datasets/samples/lending_club/`).
Dataset © 2019 Databricks, Inc., licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).