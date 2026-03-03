import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
df = pd.read_csv("data/processed/telco_features.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# Baseline Logistic Regression
baseline = LogisticRegression(max_iter=1000)
evaluate(baseline, "Logistic Regression (Baseline)")

# Class-weighted Logistic Regression
weighted = LogisticRegression(max_iter=1000, class_weight="balanced")
evaluate(weighted, "Logistic Regression (class_weight='balanced')")