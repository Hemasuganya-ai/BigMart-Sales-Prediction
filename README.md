# BigMart Sales Prediction

End-to-end data science project on the [BigMart Sales dataset](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/) — 2013 sales data for 1,559 products across 10 outlets. The project moves from raw data to a deployed, explainable prediction pipeline in four stages: **EDA & cleaning → predictive modeling → ensembles & tuning → an LLM-powered explanation layer.**

## Project structure

```
├── PART1/   # Data cleaning & exploratory data analysis
├── PART2/   # Regression + classification models
├── PART3/   # Ensembles, hyperparameter tuning, reproducible pipeline
├── PART4/   # LLM-powered model explanation feature
└── train_bigmart.csv   # Raw dataset
```

Each `PART` folder contains its own notebook/script and README with full details, formulas, and interpretations. This file gives the high-level overview.

---

## PART 1 — Data Cleaning & EDA

Raw BigMart data is inspected and cleaned: missing values in `Item_Weight` and `Outlet_Size` are imputed, inconsistent `Item_Fat_Content` labels (`LF`, `low fat`, `reg`, etc.) are standardized, and outliers/zero-visibility entries are handled. Output: `cleaned_data.csv`, used as the input for every part that follows.

## PART 2 — Predictive Modeling

Two supervised models built on the cleaned data:

- **Regression** — predicts `Item_Outlet_Sales` (continuous). Linear Regression and Ridge Regression, with coefficient interpretation and a leak-free train/test split (`StandardScaler` fit only on training data).
- **Classification** — predicts whether an item is an above- or below-median seller (`y_clf`, binarized at the median of `Item_Outlet_Sales`). Logistic Regression with class-imbalance handling, full metric suite (confusion matrix, precision/recall/F1, ROC-AUC), a decision-threshold sensitivity sweep, a regularization (`C`) experiment, and a bootstrap confidence interval on the AUC difference between regularization strengths.

**Key results:** Linear/Ridge Regression R² ≈ 0.58; Logistic Regression test AUC ≈ 0.895, accuracy ≈ 0.81.

## PART 3 — Ensembles, Tuning & a Reproducible Pipeline

Builds on Part 2's train/test split to compare model families and formalize the best one into a deployable pipeline:

- Decision Trees (unconstrained vs. depth/leaf-constrained) to illustrate the bias–variance tradeoff
- Gini vs. Entropy split-criterion comparison
- Random Forest and Gradient Boosting, with feature-importance ranking and a feature-ablation study
- 5-fold cross-validated comparison across all classifiers
- `GridSearchCV` over a full `sklearn.Pipeline` (imputer → scaler → Random Forest)
- A manual learning curve (20%–100% of training data) to diagnose whether the model is data-limited or capacity-limited
- The tuned pipeline is serialized to `best_model.pkl` and verified with a reload-and-predict check

**Key results:** Logistic Regression (from Part 2) remains the strongest and most consistent model overall (CV mean AUC ≈ 0.897, test AUC ≈ 0.895), essentially tying Gradient Boosting while being simpler and more interpretable — the recommended model for this dataset.

## PART 4 — LLM-Powered Feature

A prediction-explanation layer on top of the Part 3 model: for a given feature vector, the pipeline calls `.predict()` / `.predict_proba()`, then sends the features, predicted class, and probability to an LLM API, which returns a structured JSON explanation (predicted label, confidence, top reasons, recommended next step). Includes:

- A reusable `call_llm()` function (API key read from an environment variable, never hardcoded)
- Zero-shot prompt design with `temperature=0` for reproducible structured output
- A temperature 0 vs. 0.7 comparison to illustrate deterministic vs. sampled generation
- JSON-schema validation of every LLM response, with logged fallbacks on failure
- A regex-based PII guardrail that blocks any prompt containing an email or phone number before it reaches the API

---

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `joblib` · `requests` · `jsonschema`

## Setup

```bash
git clone https://github.com/Hemasuganya-ai/BigMart-Sales-Prediction.git
cd BigMart-Sales-Prediction
pip install pandas numpy scikit-learn matplotlib joblib requests jsonschema
```

Each `PART` folder can be run independently (Part 3 and Part 4 rebuild the necessary preprocessing from `train_bigmart.csv` / `cleaned_data.csv` at the top of their scripts). For Part 4, set an API key before running:

```bash
export LLM_API_KEY="your-key-here"
```

## Results summary

| Model | CV Mean AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.897 | 0.895 |
| Decision Tree (max_depth=5) | 0.872 | 0.848 |
| Random Forest | 0.890 | 0.890 |
| Gradient Boosting | 0.896 | 0.894 |
| Tuned Random Forest (GridSearchCV) | 0.892 | 0.890 |

**Recommended model: Logistic Regression** — best or tied-best on every metric, and the simplest and most interpretable of the group.

## License

This project is for educational/portfolio purposes, built on the publicly available BigMart Sales dataset (Analytics Vidhya practice problem).
