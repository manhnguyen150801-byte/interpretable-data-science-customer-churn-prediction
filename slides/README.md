# slides/

This folder contains the project presentation deck.

## File

| File | Description |
|---|---|
| `Slide.pdf` | Final project presentation — summary of methodology, results, interpretability analysis, and business recommendations |

---

## Presentation Overview

**Title:** Customer Churn Prediction — Interpretable Data Science Project
**Authors:** Duc Manh Nguyen, Minh Hoan Tran, Qiushuang Liu

The deck is structured as an executive-facing summary of the full analysis in `notebook/Notebook.ipynb`. It is designed to communicate findings and recommendations to a non-technical business audience without requiring code.

### Slide Sections

| Section | Content |
|---|---|
| **Business Problem & Approach** | Why churn costs matter; the 25% budget constraint; why interpretability is non-negotiable |
| **Data Insights & Multicollinearity** | Key EDA findings; VIF analysis results; treatment decision and residual risk disclosure |
| **Model Comparison Results** | Validation and test set metrics for all 5 models; interpretability tax quantified |
| **Interpretability — Intrinsic Methods** | LR coefficient analysis; key positive and protective factors |
| **Interpretability — Global Post-hoc Methods** | PDP, ICE, ALE, and SHAP beeswarm applied to XGBoost |
| **Interpretability — Local Explanations** | LIME and SHAP waterfall for the highest-risk customer in the validation set |
| **Recommendation & Business Actions** | Final model recommendation table; 4 actionable recommendations for the retention team |
| **Conclusion** | Summary of deliverables: multicollinearity treatment, model comparison, full interpretability suite |

---

## Highlighted Findings from the Deck

- **Interpretability tax:** Tuned LR vs. XGBoost gap is ~10 pp in Precision@Top25% on the test set — meaningful, but the interpretability gain justifies it for operational use
- **Decision Tree as alternative:** Only a 2.4 pp gap vs. XGBoost; IF-THEN rules are fully readable by business agents
- **Local explanation example:** The highest-risk customer (churn probability from XGBoost) has 61 service outages — addressing service quality is more effective than increasing the already-applied 17.7% discount
