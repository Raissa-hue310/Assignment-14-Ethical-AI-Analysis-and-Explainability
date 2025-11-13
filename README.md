# Assignment-14-Ethical-AI-Analysis-and-Explainability
Building, analyzing, and interpreting a machine learning model with an emphasis on fairness and explainability

This project implements a binary classification model to predict whether an individual's income is greater than \$50,000 using the Adult Income dataset. The focus is on **ethical AI**, including **fairness analysis** and **model explainability**.

## Project Overview

- **Model**: Logistic Regression (scikit-learn)
- **Dataset**: Adult Income dataset (UCI) loaded via `fetch_openml("adult", version=2)`
- **Sensitive attribute**: `sex` (Male/Female)
- **Fairness tools**: [Fairlearn](https://fairlearn.org/)
- **Explainability tools**: [SHAP](https://shap.readthedocs.io/) and [LIME](https://github.com/marcotcr/lime)

The notebook:
- Preprocesses the dataset (handling missing values, encoding categorical features, scaling numeric features)
- Trains and evaluates a logistic regression model
- Computes fairness metrics across groups defined by the sensitive feature
- Generates global (SHAP summary) and local (SHAP waterfall, LIME) explanations

## Files

- `Assignment14_ethical_ai_income.ipynb` – main Jupyter/Colab Notebook with all code and outputs
- `Assignment 14_Report.pdf` – 3-page report summarizing methodology, results, and ethical discussion
- `README.md` – this file

## Requirements

Install the following Python packages:

```bash
pip install numpy pandas scikit-learn fairlearn shap lime matplotlib seaborn
