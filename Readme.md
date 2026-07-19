# Credit Loan Default Prediction

## Overview
Systematic comparison of class imbalance techniques on Home Credit dataset (307,511 samples, 92% vs 8% imbalance).

## Key Finding
Built-in class balancing (`scale_pos_weight`) significantly outperforms SMOTE for XGBoost 
(64.4% vs 45.4% recall, p<0.000001) — a counterintuitive result challenging conventional wisdom.

## Approaches Compared
| Approach | XGBoost Recall | Description |
|----------|---------------|-------------|
| Baseline | 2.2% | No imbalance handling |
| Built-in Balancing | **64.4%** | `scale_pos_weight` |
| SMOTE + PR Tuning | 45.4% | Synthetic oversampling + threshold optimization |

## Statistical Validation
- McNemar's exact test: all pairwise p < 0.000001
- 5-fold CV: 64.1% ± 0.7% recall, 75.5% ± 0.3% ROC-AUC

## Explainability
SHAP analysis reveals top predictors: external credit scores, debt-to-income ratio, borrower age

## Tech Stack
Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn