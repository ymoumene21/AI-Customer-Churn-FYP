import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Load dataset
df = pd.read_csv("data/processed/telco_features.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Thresholds to test
thresholds = [0.3, 0.4, 0.5, 0.6]

rows = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    rows.append({
        "threshold": t,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    })

results = pd.DataFrame(rows)

# Save for report
os.makedirs("reports/business_eval", exist_ok=True)
out_path = "reports/business_eval/threshold_summary.csv"
results.to_csv(out_path, index=False)

print("Saved threshold summary to:", out_path)
print("\nThreshold Summary:")
print(results.to_string(index=False))

# Simple cost framing (edit values if needed)
COST_FN = 10  # e.g., lost customer cost
COST_FP = 1   # e.g., unnecessary retention contact cost

results["estimated_cost"] = results["FN"] * COST_FN + results["FP"] * COST_FP
results = results.sort_values("estimated_cost")

out_cost = "reports/business_eval/threshold_cost_ranking.csv"
results.to_csv(out_cost, index=False)

print("\nSaved cost ranking to:", out_cost)
print("\nCost ranking (lower is better) using COST_FN=10, COST_FP=1:")
print(results[["threshold", "FN", "FP", "estimated_cost"]].to_string(index=False))