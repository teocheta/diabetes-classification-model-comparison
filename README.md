# Diabetes Classification — Model Comparison

End-to-end machine learning pipeline for diabetes prediction using structured medical data.
Includes preprocessing, training, evaluation, and model comparison.

## Features
- Data cleaning: invalid zeros treated as missing values
- Preprocessing: median imputation + StandardScaler
- Models: Logistic Regression, KNN, Random Forest, XGBoost
- Evaluation: accuracy/precision/recall/F1 + classification report + confusion matrix
- CLI demo: interactive patient input and prediction

## Project Structure
- `main.py` (or your entry file): runs evaluation + demo menu
- `models/`: model implementations and evaluation utilities
- `outputs/`: generated metrics and plots (not committed)

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
