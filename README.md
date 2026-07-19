# Employee Attrition Prediction

This Machine Learning project predicts employee attrition (voluntary departure) from HR analytics data covering demographics, compensation, job satisfaction, and career-history attributes. The task is formulated as a binary classification problem: predicting whether an employee's `Attrition` status is `Yes` or `No`.

The project follows a complete ML workflow: dataset provenance and signal-validation, exploratory data analysis, leakage-safe feature engineering and preprocessing, baseline model training, hyperparameter tuning validated against resampling noise, decision-threshold and cost-sensitive optimization, probability calibration, and production-pipeline assembly.

## Objectives

- Build a machine learning model to predict whether an employee will leave the company (`Attrition = Yes`).
- Identify which factors — compensation, overtime, tenure, business travel, marital status, job role — are most associated with attrition, and validate those associations statistically before relying on them.
- Compare five classification approaches: Decision Tree, K-Nearest Neighbors, Logistic Regression, Random Forest, and Gradient Boosting.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC on the minority (`Attrition = Yes`) class, given the ~84/16 class imbalance.
- Select a decision threshold appropriate to the retention use case (F-beta and cost-ratio criteria) and a capacity-constrained top-K targeting alternative.
- Check whether hyperparameter tuning's effect on test performance is real or an artifact of a single train/test split, using a repeated-split robustness check.

## Dataset

The dataset is the [`saadharoon27/hr-analytics-dataset`](https://www.kaggle.com/datasets/saadharoon27/hr-analytics-dataset) on Kaggle, a redistribution of IBM's *HR Analytics Employee Attrition & Performance* dataset (1,480 rows, 38 columns). The notebook downloads it directly via `kagglehub`.

**Important caveat.** IBM documents this dataset as fictional, constructed for a Watson Analytics demo rather than sourced from real payroll records. The notebook's first section evaluates whether it nonetheless carries genuine statistical structure (as opposed to placeholder data with no real feature-target relationship) before proceeding to model it; results should not be interpreted as describing real employee outcomes.

Key columns include `Attrition` (target), `MonthlyIncome`, `OverTime`, `YearsAtCompany`, `JobRole`, `BusinessTravel`, `MaritalStatus`, and five ordinal satisfaction/involvement scores.

## Project Workflow

The notebook is organized into the following stages:

- **Dataset Provenance and Signal Validation**: checks row-level integrity, degenerate columns, missingness pattern, and statistical association with the target, to confirm the dataset carries genuine signal before modeling it.
- **Data Cleaning**: drops identifier and constant columns, removes duplicate rows, fixes a `BusinessTravel` data-entry inconsistency.
- **Exploratory Data Analysis**: target distribution, income/tenure/overtime relationships, attrition rate by job role, tenure, business travel, marital status × gender, and promotion recency.
- **Train/Test Split**: fixed early, before any feature-selection decision, and reused unchanged through the rest of the notebook.
- **Feature Engineering**: six candidate engineered features (job-hopping ratio, manager-stability ratio, satisfaction composite, new-hire flag, promotion-stagnation ratio, income-relative-to-job-level) are tested for significance on the training partition only, and four are retained.
- **Baseline Preprocessing**: leakage-safe imputation, one-hot encoding, and scaling fit on the training partition only.
- **Baseline Model Training**: five classifiers compared on identical processed features.
- **Hyperparameter Tuning**: randomized/grid search (5-fold stratified CV) for Random Forest, Logistic Regression, and Gradient Boosting.
- **Repeated-Split Robustness Check**: the single-split tuning result is re-tested across 20 independent train/test splits with a paired significance test, to separate a real tuning effect from resampling noise.
- **Decision Threshold Tuning**: F1-optimal threshold via out-of-fold predictions, then F-beta (F1/F1.5/F2) and cost-ratio criteria across all tuned models.
- **Probability Calibration Check**: reliability curves and Brier scores, since the cost-ratio threshold assumes calibrated probabilities.
- **Capacity-Constrained Targeting (Top-K)**: ranks employees by predicted risk for a fixed-capacity retention program.
- **Feature Importance**: impurity-based importances (Random Forest, Gradient Boosting), standardized coefficients (Logistic Regression), and SHAP values for the two tree models, cross-checked against each other.
- **Production Pipeline**: bundles preprocessing and each tuned model into a single deployable `sklearn.Pipeline`, persisted with `joblib`.

## Models

The following classifiers were compared:

- Decision Tree
- K-Nearest Neighbors
- Logistic Regression
- Random Forest
- Gradient Boosting

No single model is best across every use case. Logistic Regression (tuned) gives the most consistent precision/recall balance under F-beta thresholds; Gradient Boosting (tuned) gives the most precise ranking for capacity-constrained top-K targeting (100% precision at K=20); Random Forest (tuned) sits between the two. A repeated-split robustness check confirmed tuning's effect is statistically real for Gradient Boosting, but not distinguishable from resampling noise for Random Forest or Logistic Regression.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
└── Machine_Learning_Project.ipynb
```

The dataset and the trained pipeline artifacts (`attrition_pipeline_*.joblib`) are not stored in the repository; the notebook downloads the dataset via `kagglehub` and regenerates the pipelines on each run.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open and run the notebook:

```text
Machine_Learning_Project.ipynb
```

The dataset is downloaded inside the notebook using `kagglehub`, so Kaggle access may be required.

## Results Summary

Baseline models, default threshold, test set (n=295, 47 positive):

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.79 | 0.43 | 0.89 | 0.58 | 0.897 |
| Gradient Boosting | 0.89 | 0.73 | 0.47 | 0.57 | 0.869 |
| Random Forest | 0.88 | 0.61 | 0.66 | 0.63 | 0.858 |
| KNN | 0.86 | 1.00 | 0.11 | 0.19 | 0.798 |
| Decision Tree | 0.80 | 0.41 | 0.60 | 0.49 | 0.656 |

After tuning, the three strongest models trade off differently depending on the deployment mode, so no single leaderboard ranking applies:

| Model (tuned) | Test ROC-AUC | Best F1 (F-beta search) | Precision @ K=20 |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.899 | 0.667 | 0.80 |
| Gradient Boosting | 0.867 | 0.638 | 1.00 |
| Random Forest | 0.852 | 0.588 | 0.70 |

Overtime and below-median income are the strongest individual predictors; tenure shows a pronounced early-career effect (34.9% attrition in year one, down to 8.1% after ten years); frequent business travel and single marital status are additional risk factors. Two engineered features, job-hopping ratio and a satisfaction composite, rank among the top five predictors by importance. SHAP values on the two tree models surfaced `StockOptionLevel` as a protective predictor (top 5 by mean |SHAP| in both) that the impurity-based importances had ranked well outside the top 10 — impurity importance systematically underrates one-hot and ordinal-int features relative to continuous ones.

## Future Work

Several improvements could be explored in future work:

- Add domain-plausibility / cross-field consistency checks to the data-quality stage (e.g. `Age` vs. `TotalWorkingYears`, `YearsInCurrentRole` vs. `YearsAtCompany`), beyond the statistical signal-validation already performed.
- Test whether `YearsWithCurrManager` missingness is itself associated with attrition (a missing-not-at-random check) rather than assuming it is missing at random.
- Use mutual information alongside Welch's t-test / chi-square to catch non-linear feature-target associations that a mean-difference test can miss.
- Apply `CalibratedClassifierCV` before using the cost-ratio threshold operationally, particularly for the class-weighted logistic regression model, which the calibration check found to be overconfident.
- Resolve the fairness question flagged in the notebook's conclusion: `Gender` and `MaritalStatus` are both meaningful predictors in the tuned logistic regression model, which raises disparate-impact considerations that would need an explicit decision before any real-world deployment.
- Re-validate on real (non-fictional) HR data before drawing organizational conclusions, since IBM documents this dataset as a Watson Analytics demo rather than real payroll records.
