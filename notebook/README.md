# notebook/

This folder contains the main analysis notebook for the project.

## File

| File | Description |
|---|---|
| `Notebook.ipynb` | End-to-end churn prediction analysis — EDA, preprocessing, modelling, interpretability, and business recommendations |

---

## Notebook Structure

The notebook is written as a **professional consulting report with embedded code**. Every code cell is preceded by a markdown cell explaining *what* it does and *why*. Results are always interpreted, not just displayed.

| Section | Content |
|---|---|
| **1. Introduction** | Business context, problem statement, why interpretability matters |
| **2. Setup & Imports** | Library imports, random seed (`RANDOM_STATE = 42`) |
| **3. EDA** | Data overview, missing value audit, class balance (27% churn), feature distributions, binary feature churn rates, correlation matrix, VIF analysis |
| **4. Preprocessing** | Multicollinearity treatment (3 features dropped), feature engineering (3 new features), one-hot encoding of `state`, 60/20/20 stratified split, StandardScaler |
| **5. Baseline Model** | Default Logistic Regression — ROC-AUC 0.713, Precision@Top25% 63.6% on validation |
| **6. Model Development** | GridSearchCV tuning of Logistic Regression, Decision Tree, Random Forest, and XGBoost; validation-set comparison table |
| **7. Interpretability** | LR coefficients, Decision Tree visualisation, RF vs XGBoost feature importance, PDP, ICE, ALE, Global Surrogate, LIME, SHAP (global + local) |
| **8. Multicollinearity** | Full before/after VIF comparison; treatment decision rationale |
| **9. Conclusion** | Final model selection (Tuned LR), test-set evaluation, business recommendations |

---

## Outputs

Running the notebook end-to-end generates 15 chart files saved to `../outputs/charts/`. The notebook reads data from `../data/churn_2024.csv`.

---

## Reproducibility

All random operations use `RANDOM_STATE = 42`. Run all cells top-to-bottom in a fresh kernel to reproduce every result exactly.
