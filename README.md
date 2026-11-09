# Credit Card Fraud Detection and Risk Analysis (ML)

A machine learning project that detects fraudulent credit card transactions and scores the risk of each transaction, so suspicious activity can be flagged for review.

> **Status:** Work in progress. This README describes the project goals and planned structure. Results will be added once the models are trained and evaluated.

## Overview

Credit card fraud is rare compared to genuine transactions, which makes it a highly imbalanced classification problem. This project aims to:

- Explore and understand transaction data (EDA)
- Handle class imbalance (e.g. SMOTE, undersampling, class weights)
- Train and compare classification models
- Evaluate with metrics suited to imbalanced data
- Convert model output into a risk score / risk category

## Approach

1. **Data collection** - transaction dataset (e.g. the public Kaggle *Credit Card Fraud Detection* dataset)
2. **Preprocessing** - cleaning, scaling of `Amount` and `Time`, train/test split
3. **Imbalance handling** - resampling and/or class weighting
4. **Modeling** - Logistic Regression, Random Forest, XGBoost and similar models
5. **Evaluation** - Precision, Recall, F1-score, ROC-AUC and PR-AUC (accuracy alone is misleading here)
6. **Risk analysis** - map predicted fraud probability to Low / Medium / High risk

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn, imbalanced-learn, XGBoost
- matplotlib, seaborn
- Jupyter Notebook

## Planned Project Structure

```
.
├── data/            # dataset (not committed if large)
├── notebooks/       # EDA and model experiments
├── src/             # preprocessing, training, evaluation code
├── models/          # saved models
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/Mrr-Sandy/-Credit-Card-Fraud-Detection-and-Risk-Analysis-ML.git
cd -Credit-Card-Fraud-Detection-and-Risk-Analysis-ML
pip install -r requirements.txt
jupyter notebook
```

(`requirements.txt` and notebooks will be added as the project develops.)

## Results

To be added: model comparison table, confusion matrix and ROC / PR curves.

## Author

[Mrr-Sandy](https://github.com/Mrr-Sandy)
