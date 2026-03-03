import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/processed/telco_features.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# Model (same baseline)
# -----------------------------
model = LogisticRegression(max_iter=1000)

# -----------------------------
# 5-fold Stratified CV
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)

def summarize(metric_name: str):
    scores = results[f"test_{metric_name}"]
    return np.mean(scores), np.std(scores)

print("Logistic Regression - 5-Fold Stratified Cross-Validation")
print("--------------------------------------------------------")

for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    mean, std = summarize(m)
    print(f"{m.upper():9s}: {mean:.4f} ± {std:.4f}")