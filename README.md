# Diamond Price Prediction

Predicting diamond prices (USD) from physical and quality attributes using scikit-learn, with a linear regression baseline, a Random Forest model, and SHAP-based explainability.

## Overview

The notebook walks through a complete regression workflow: exploratory analysis, feature encoding, model training, evaluation, and interpretation. A Random Forest reduces test error by more than half compared to the linear baseline.

## Dataset

The Diamonds dataset (`Price.csv`), one row per diamond.

| Feature | Description |
|---|---|
| Carat | Weight of the diamond |
| Cut | Cut quality (Fair, Good, Very Good, Premium, Ideal) |
| Color | Grade from D (best) to J (worst) |
| Clarity | Clarity grade (I1 to IF) |
| Depth, Table | Proportion measurements (%) |
| X, Y, Z | Length, width, depth (mm) |
| **Price** | **Target, in US dollars** |

## Approach

1. **EDA:** price distribution (right-skewed), carat vs. price relationship, cut and clarity counts
2. **Preprocessing:** one-hot encoding for `Cut` and `Color`, label encoding for `Clarity`
3. **Split:** 70% train / 30% test, `random_state=42`
4. **Models:** Linear Regression (baseline) and Random Forest (`n_estimators=100`, `max_depth=10`, `min_samples_split=3`)
5. **Evaluation:** R², RMSE, MAE on train and test sets, plus actual-vs-predicted and residual plots
6. **Explainability:** feature importances and SHAP summary plot

## Results (test set)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.893 | $1,294 | $827 |
| **Random Forest** | **0.976** | **$608** | **$332** |

The Random Forest train/test R² gap is small (0.981 vs. 0.976), so overfitting is limited.

## Key findings

- **Carat dominates.** It is the top feature in both models (Random Forest importance: 0.63), followed by diamond width (0.27) and clarity (0.06).
- **Linear regression struggles with the non-linear price curve.** Price grows faster than linearly with size, which trees capture and a linear model cannot.
- **Colour matters, cut less so.** Colour J (lowest grade) carries the largest negative linear coefficient among the categorical features.

