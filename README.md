# Bank GoodCredit – Credit Risk Scoring

A credit risk project for a bank client. The goal is to predict which credit card customers are likely to miss payments, so the bank can reduce default losses.

**Project ID:** PM-PR-0015 | **Domain:** Banking – Credit Risk | **Tools:** Python, MySQL, scikit-learn, XGBoost.

---

## Business Problem

Bank GoodCredit wants a credit score for its current credit card customers. The score should show how trustworthy a customer is, and rank customers from low risk to high risk.

- **Target:** `Bad_label`
  - `0` = good credit history
  - `1` = bad credit history (30 or more days past due)
- **Bank's benchmark:** Gini = 37.9

## Data

The data comes from a MySQL database with three tables:

| Table | What it has | Rows per customer |
|---|---|---|
| `Cust_Account` | Past accounts and payment history | Many |
| `Cust_Enquiry` | Credit enquiries (date, purpose, amount) | Many |
| `Cust_Demographics` | Current applications, 79 hidden features and `Bad_label` | One |

The data is confidential client data, so it is **not included** in this repository. The 79 demographic features are renamed by the bank, so their meaning is unknown.

## Approach

1. **Loaded** the three tables from MySQL.
2. **Explored** each table: missing values, duplicates, skew, outliers and correlations.
3. **Aggregated** the account and enquiry tables to one row per customer (account count, balances, utilisation, enquiry counts in the last 90 and 365 days), then merged them with the demographics table.
4. **Preprocessed** with a pipeline: outlier capping, power transform, encoding and scaling, all fitted on the training data only.
5. **Handled class imbalance** with SMOTE inside the cross-validation pipeline. Only about 3.6% of the rows are bad.
6. **Trained and compared** Logistic Regression, Decision Tree, Random Forest, XGBoost and an ANN.
7. **Evaluated** with Gini, ROC AUC and decile rank ordering, not accuracy.

## Results

<!-- Fill these in after the rebuild. Don't write numbers you have not run. -->

| Metric | Bank benchmark | My model |
|---|---|---|
| Gini | 37.9 | _TBD_ |

**Decile rank ordering (bad rate per decile):** _add table here_

**Top features by gain:** _add table here_

## What I Learned

- Accuracy is misleading when only 3.6% of the data is bad. Gini and rank ordering fit the business goal better.
- Account and enquiry data must be combined to one row per customer before modelling. Splitting account-level rows randomly leaks the same customer into train and test.
- Preprocessing and resampling must happen inside the training folds only, or the scores look better than they really are.

## Limitations

- The demographic features are hidden, so they are hard to interpret.
- The time between the application and the `Bad_label` outcome is not given, so I could not check the time window.
- The model needs calibration and monitoring before any real use.

## How to Run

1. Install the libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn pymysql matplotlib seaborn
   ```
2. Set your own database details as environment variables. Never write them in the notebook.
   ```bash
   export DB_HOST=your_host
   export DB_USER=your_user
   export DB_PASSWORD=your_password
   ```
3. Open `Bank_Good_Credit.ipynb` and run the cells in order.

## Author

**Darshit** – Data Science
LinkedIn : www.linkedin.com/in/darshit-rathod-20712922b | GitHub : https://github.com/darshitrathod16
