# 🏦 Loan Default Prediction

**A personal machine learning project, built end to end by me on Databricks**

`Databricks` · `Python` · `PySpark` · `Spark SQL` · `Spark ML` · `MLflow` · `Unity Catalog` · `Delta Lake`

**Author:** Shafin · GitHub: [@Shafinzz](https://github.com/Shafinzz)

---

## 📌 Table of contents
1. [What is this project about?](#-1-what-is-this-project-about)
2. [The problem](#-2-the-problem)
3. [What needed to be done](#-3-what-needed-to-be-done)
4. [What I did, step by step](#-4-what-i-did-step-by-step)
5. [Final outcome](#-5-final-outcome)
6. [How to run it](#-6-how-to-run-it)
7. [Limitations and next steps](#-7-limitations-and-next-steps)
8. [What I learned](#-8-what-i-learned)
9. [Simple glossary](#-9-simple-glossary)
10. [Data and credit](#-10-data-and-credit)

---

## 🧭 1. What is this project about?

Banks and lending companies lose money when people **borrow money and do not pay it back**. This is called a
**default**.

In this project, I built a **machine learning model** that looks at a loan application and predicts
**how likely the person is to default**. The lender can then use this risk score to decide whether to approve
or reject the loan.

This is my **own personal project**. I explored the data, made the decisions, wrote and ran every notebook,
tracked every experiment, and documented the results myself.

**In one sentence:** I used 1.48 million real loans from Lending Club to build, test and register a model that
spots risky loans better than the simple rule the lender used before.

---

## 💼 2. The problem

- A peer-to-peer lending company gives out **thousands of personal loans every month**.
- About **1 in 5 finished loans ends in default** (20.6% in this data). That costs the company **millions of dollars**.
- Right now, the company uses **one simple rule**: *"reject anyone with a low credit score."*
- That rule is too simple:
  - ❌ it **rejects good people** who would have paid back
  - ❌ it **still approves people** who later stop paying

**The goal:** replace the simple rule with a model that uses **the whole application** (income, debt, loan size,
loan length, job history, credit history...) and **prove with numbers** that it works better.

---

## ✅ 3. What needed to be done

The project had 5 main parts. Each one is a notebook in this repo.

| # | Part | What it required |
|---|---|---|
| 1 | **Explore the data** (EDA) | Check data quality, find "cheating" columns (data leakage), show how many loans default, find patterns with charts |
| 2 | **Prepare the data** | Fix messy values, fill missing values, turn text into numbers, build a reusable **Spark ML Pipeline**, split the data by time |
| 3 | **Train models** | Train **at least 2 model types**, tune settings, handle the fact that defaults are rare, log **at least 6 experiments in MLflow** |
| 4 | **Evaluate** | Use the right scores (not just accuracy), explain mistakes in money terms, compare against the simple rule, show what drives defaults |
| 5 | **Register the model** | Save the best model in the **Model Registry**, prove it can score new loans, write a **model card** |

**Rules I followed:** PySpark and Spark ML only (no scikit-learn), every experiment tracked in MLflow,
and every decision explained.

---

## 🛠️ 4. What I did, step by step

### Step 0: Setup (`00_config`)
- Created one **settings notebook** with all paths, table names, feature lists and the MLflow experiment name.
- Every other notebook loads it with `%run ./00_config`, so names are written **only once**.
- Created my own schema in Unity Catalog: `workspace.loan_default`.

---

### Step 1: Exploring the data (`01_eda`)

#### 🔎 Part A: I checked the suggested data, and replaced it
The project suggested a file called `LoanStats_2018Q2.csv` (**130,772 loans**, **145 columns**).
Before using it, I checked it carefully:

- **93.65%** of the loans were still **"Current"**, meaning people were still paying, so **nobody knew yet** if they would default.
- Only **5,794 loans** had finished, with only **134 defaults**. That's far too few to teach a model.
- I also found **4 junk rows** in the file (2 empty lines and 2 summary lines), which I removed.

So I looked for a better source and found a larger Lending Club dataset (**2012–2017**) already built into Databricks:

| | Suggested file (2018 Q2) | Chosen dataset (2012–2017) |
|---|---|---|
| Total loans | 130,772 | **1,481,541** |
| Loans with a known ending | 5,794 | **658,049** (113× more) |
| Defaults to learn from | 134 | **135,724** |
| Default rate (finished loans) | 2.3% | **20.6%** |

I also removed **19 broken rows** from the new dataset (missing status, or values shifted into the wrong column).

#### 🚫 Data leakage: removing "cheating" columns
Some columns describe what happened **after** the loan was given, like *"total amount paid back"* or *"money recovered after default"*.
A model that uses them would look amazing in testing but be **useless in real life**, because on the day someone applies, nobody knows these values yet.

My test for every column: **"Would the lender know this on the day the person applies?"**
- ✅ Yes → safe to use
- ❌ No → leakage → removed

I found and removed **29 leakage columns** (payments, outstanding balance, recoveries, hardship plans, last payment dates).

#### ⚖️ Class imbalance
Among finished loans: **79.4% repaid** and **20.6% defaulted**.
A lazy model that always says "repaid" would be **79% correct** but catch **zero defaults**.
So I did **not** use accuracy to judge the model.

#### 🔍 Missing credit score
The data has **no FICO credit score**. So the old rule ("reject if credit score is below 650") could not be applied.
I replaced it with an equally simple rule using Lending Club's own **grade** (A = safest, G = riskiest).

#### 📊 Patterns I found
| Feature | What I found |
|---|---|
| **Grade** | Default rate climbs steadily: **A ≈ 6.5%**, B ≈ 13%, C ≈ 22%, D ≈ 30%, E ≈ 39%, F ≈ 45%, **G ≈ 50%**. A grade G loan is about **7× riskier** than grade A |
| **Loan length** | **60-month** loans default about **twice as often** as 36-month loans (**~33.5% vs ~16.5%**) |

#### 🧹 Data quality report (the problems I found and fixed later)
| Problem | Example | How many | Fix |
|---|---|---|---|
| Numbers stored as text | `" 36 months"`, `"  5.32%"`, `"96.3%"` (with hidden spaces!) | 5 columns | Remove the extra characters and convert to numbers |
| Hidden missing values | Job length written as `"n/a"` instead of empty | **33,672 loans (5.1%)** | Treat as missing and add a "job length unknown" flag |
| Impossible values | Debt-to-income of **-1** or **999**, income of **$0** | ~53 loans | Treat as missing |
| Missing values | Mortgage accounts empty | **7,495 loans (1.14%)** | Fill with the middle value (median) |
| Very rare categories | Home ownership "ANY", "NONE", "OTHER" | **131 loans** | Merge into one "OTHER" group |

#### 🎯 Columns I chose: 20
Loan amount, loan length, interest rate, monthly payment, grade, loan purpose, job length, home ownership,
yearly income, income verified or not, debt-to-income, late payments in the last 2 years, recent credit checks,
open accounts, public records, credit card balance, credit card usage %, total accounts, mortgage accounts, bankruptcies.

**Left out on purpose:** location (state and zip code), so the model does not treat people differently based on
where they live; duplicate columns; and mostly-empty columns.

---

### Step 2: Preparing the data (`02_feature_engineering`)

**1. The answer column (target):**
- Fully Paid → **0** (good)
- Charged Off / Default → **1** (bad)
- Loans **still running** (Current, Late, Grace Period) → **removed**, because their ending is unknown

**2. A cleaning machine (Spark ML Pipeline) with 5 stages:**

| Stage | Tool | What it does |
|---|---|---|
| 1 | `SQLTransformer` | Cleans the text numbers, fixes impossible values, merges rare groups, adds the "job length unknown" flag |
| 2 | `Imputer` | Fills missing values with the **median** (safe against extreme values like an $8.9M income) |
| 3 | `StringIndexer` | Turns words (like "RENT") into number codes |
| 4 | `OneHotEncoder` | Turns those codes into on/off switches |
| 5 | `VectorAssembler` | Packs everything into **one list of 45 numbers** per loan |

**Why a Pipeline?** A new application gets cleaned **exactly the same way** as the training data, automatically, with one command.

**3. Splitting by time (not randomly):**
The model learns from **older** loans and is tested on **newer** loans, just like real life.

| Set | Years | Loans | Default rate |
|---|---|---|---|
| Training | 2012–2015 | **547,821** | 20.76% |
| Test | 2016 | **98,301** | 21.96% |
| Not used | 2017 | 11,927 | 3.56%: too new, mostly early payoffs |

The cleaning machine learned its values (like the medians) **from the training data only**, so no test data leaked in.

---

### Step 3: Training the models (`03_ml_training`)

**Handling rare defaults: class weights.**
There are **434,107 repaid** vs **113,714 defaulted** loans in training. So each default was given a weight of
**3.82**, which makes both groups equally important. I chose this instead of copying default rows (which can cause
overfitting) or deleting repaid rows (which wastes real data).

**Fair tuning with a "practice exam":**
To compare settings, I trained on **2012–2014** (**362,148** loans) and checked on **2015** (**185,673** loans).
The 2016 test set stayed **untouched** until the final evaluation.

**9 experiments, all tracked in MLflow** (settings, data years, imbalance method, and scores):

| Model | Settings tried | Best AUC-ROC | Defaults caught (recall) |
|---|---|---|---|
| Logistic Regression | 3 values of `regParam` (0, 0.01, 0.1) | 0.711 | 56.5% |
| Random Forest (50 trees) | 3 tree depths (5, 8, 10) | 0.706 | 56.1% |
| **Gradient-Boosted Trees** | 50 rounds, depth 5 | **0.712** 🏆 | **60.4%** 🏆 |

**Winner: Gradient-Boosted Trees (GBT).** It had the best scores, especially at catching defaults.
I then retrained it on **all training years (2012–2015)** and saved it. I also saved the Logistic Regression as a backup to compare.

---

### Step 4: Evaluating the model (`04_evaluation`)

#### 📝 Final test on 2016 loans (never seen before)
| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| **GBT** | **0.706** | 33.3% | **68.3%** | **0.448** |
| Logistic Regression | 0.703 | 33.3% | 67.3% | 0.445 |

**In plain English (GBT):**
- **AUC-ROC 0.706:** if you pick one person who defaulted and one who repaid, the model gives the defaulter the higher risk score about **71% of the time**.
- **Recall 68.3%:** it catches about **7 out of 10** people who really defaulted.
- **Precision 33.3%:** of the loans it flags as risky, about **1 in 3** really defaulted.

#### 💸 Mistakes, counted in money (cut-off 0.5)
| | Model approves | Model rejects |
|---|---|---|
| **Person repaid** | ✅ 47,235 loans ($623.9M) | ❌ 29,481 good borrowers rejected ($480.4M) |
| **Person defaulted** | ❌ 6,854 bad loans approved ($95.3M) | ✅ 14,731 bad loans stopped ($246.8M) |

- Approving a defaulter = the lender **loses the money**. This is the most expensive mistake.
- Rejecting a good borrower = the lender **loses the interest profit** and the customer.

#### 🎚️ The "strictness dial" (cut-off)
The lender can choose how strict the model is:

| Cut-off | Applicants rejected | Defaults caught | Rejected loans that really defaulted |
|---|---|---|---|
| 0.50 | 45% | 68% | 33% |
| **0.60** | **29%** | **50%** | **38%** (vs 22% if rejecting at random) |
| 0.70 | 14% | 27% | 44% |

Picking the cut-off is a **business decision**: it balances losing money against losing customers.

#### 🥊 Model vs the simple rule (fair test: same number of rejections)
| Simple rule | % rejected | Rule catches | **Model catches** |
|---|---|---|---|
| Reject grade G | 1.0% | 2.6% | **2.9%** |
| Reject grades F–G | 4.5% | 10.1% | **10.7%** |
| Reject grades E–G | 12.8% | 25.3% | **26.2%** |
| Reject grades D–G | 28.7% | 47.8% | **50.0%** |

**The model wins every time.** At about 29% rejected, it catches **~490 more defaulters per year** with the
**same number of rejections**, which is roughly **$8M** of risky loans avoided.

#### 🔑 What drives defaults (top 10)
| # | Feature | Importance |
|---|---|---|
| 1 | Interest rate | **21.8%** |
| 2 | Debt-to-income | 7.6% |
| 3 | Yearly income | 7.5% |
| 4 | Loan amount | 6.4% |
| 5 | Mortgage accounts | 6.0% |
| 6 | Loan length (term) | 4.8% |
| 7 | **Job length unknown** (a flag I created) | 4.6% |
| 8 | Open accounts | 4.4% |
| 9 | Credit card balance | 4.3% |
| 10 | Monthly payment | 3.2% |

**Main lessons:** the **interest rate** matters most (Lending Club already prices risk into it), then
**affordability** (debt, income, loan size). The **"job length unknown"** flag I engineered made the top 10.

---

### Step 5: Registering the model (`05_model_registry`)

- Registered the model in the **Unity Catalog Model Registry** as **`workspace.loan_default.loan_default_gbt`** (version 1).
- Added a full **description** (what it predicts, training data, inputs, scores).
- Gave it the alias **`@champion`**, meaning "this is the version to use".
- **Batch prediction test:** loaded the model **by name** and scored all **98,301** test loans in one go, straight from
  **raw** application data (like `"  5.32%"` and `" 36 months"`). The result matched evaluation exactly (**AUC-ROC 0.706**).
- Wrote a **model card**: what the model is for, how well it works, its limits, and what to do before using it for real.

---

## 🏁 5. Final outcome

| What | Result |
|---|---|
| Data checked | **1,481,541 loans**, 137 columns |
| Loans used for learning and testing | **646,122** (547,821 train + 98,301 test) |
| Leakage columns removed | **29** |
| Experiments tracked in MLflow | **9** (3 model types) |
| Final model | **Gradient-Boosted Trees** in a 6-stage Spark ML Pipeline |
| Test score (AUC-ROC) | **0.706** |
| Defaults caught at a 29% reject rate | **50%** |
| vs the simple grade rule | **wins at every level**; ~**490 more** defaulters caught per year |
| Registered model | `workspace.loan_default.loan_default_gbt` **@champion** |

✅ All 5 project parts completed: EDA, feature engineering, training with MLflow, business evaluation, and model registry with a model card.

---

## ▶️ 6. How to run it

**You need:** a Databricks workspace (the free **Databricks Free Edition** works, with serverless compute).

1. In Databricks, go to **Workspace → Create → Git folder** and paste this repo's URL.
2. Open the `notebooks` folder.
3. Run the notebooks **in this order**:

| Order | Notebook | What it does | Time (approx.) |
|---|---|---|---|
| 1 | `00_config` | Sets up names, schema, volume and MLflow experiment | seconds |
| 2 | `01_eda` | Explores the data and saves the main table | a few minutes |
| 3 | `02_feature_engineering` | Builds the Pipeline and saves train/test tables | a few minutes |
| 4 | `03_ml_training` | Trains 9 models and logs them in MLflow | 20–40 minutes |
| 5 | `04_evaluation` | Tests the final models and compares with the simple rule | a few minutes |
| 6 | `05_model_registry` | Registers the model, runs batch predictions, model card | a few minutes |

**What gets created:**
- Tables in `workspace.loan_default` (`loans_raw`, `train`, `test`)
- A volume `workspace.loan_default.artifacts` (saved Pipeline)
- An MLflow experiment at `/Users/<your-email>/loan_default_experiments`
- A registered model `workspace.loan_default.loan_default_gbt`

The data is **already built into Databricks**, so there's nothing to download.

---

## ⚠️ 7. Limitations and next steps

**Limitations (also in the model card):**
1. **No credit score** in the data, which is one of the strongest risk signals.
2. The model **depends a lot on the interest rate**, which already contains Lending Club's own risk pricing.
3. The data is **from 2012–2016**. People and the economy have changed, so the model would need retraining before real use.
4. The risk scores are **rankings, not exact probabilities** (class weights push them upward).
5. **Fairness has not been audited** yet.

**Next steps I would take:**
- Add credit-score or credit-bureau data
- Calibrate the probabilities
- Run a fairness check
- Set up drift monitoring and a retraining schedule
- Serve the model as a real-time API

---

## 💡 8. What I learned

- **Check the data before trusting it.** The suggested file was 94% unfinished loans; checking it first saved the project.
- **Leakage is sneaky.** The most "useful-looking" columns were the ones that see the future.
- **Accuracy can lie** when one outcome is much more common than the other.
- **Simple models can be strong.** Logistic Regression came within 0.003 AUC of the best model.
- **Every score needs a business meaning.** A cut-off is a trade-off between losing money and losing customers.
- **Track everything.** MLflow made every experiment easy to compare and the final model easy to load by name.

---

## 📖 9. Simple glossary

| Term | Simple meaning |
|---|---|
| **Default** | The borrower stopped paying the loan back |
| **Data leakage** | Using information the model would not have at prediction time (like "cheating" on a test) |
| **Class imbalance** | One outcome (repaid) is much more common than the other (default) |
| **Pipeline** | A machine with fixed steps that prepares data the same way every time |
| **AUC-ROC** | How well the model ranks risky loans above safe ones (0.5 = coin flip, 1.0 = perfect) |
| **Recall** | Of all real defaults, how many the model caught |
| **Precision** | Of all loans the model flagged, how many really defaulted |
| **Cut-off** | The risk score above which a loan is rejected |
| **MLflow** | A tool that records every experiment (settings, scores, models) |
| **Model Registry** | An official shelf for ready-to-use models, with names and versions |
| **Model card** | A short "user manual" for a model: use, performance, limits |

---

## 📜 10. Data and credit

Lending Club loan data from [Kaggle (Wendy Kan)](https://www.kaggle.com/wendykan/lending-club-loan-data),
in the Parquet version provided by Databricks as a built-in sample (`/databricks-datasets/samples/lending_club/`).
Dataset © 2019 Databricks, Inc., licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).