import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1) Load dataset (model-ready)
# -----------------------------
df = pd.read_csv("data/processed/telco_features.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# 2) Train same baseline model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------------------
# 3) Extract coefficients (feature effects)
# -----------------------------------------
feature_names = X.columns
coefs = model.coef_.ravel()

# Odds ratio = exp(coef)
odds_ratios = np.exp(coefs)

results = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs,
    "odds_ratio": odds_ratios,
    "direction": np.where(coefs > 0, "increases churn", "reduces churn"),
    "abs_coefficient": np.abs(coefs)
})

# Rank by absolute coefficient magnitude (strongest drivers)
results = results.sort_values("abs_coefficient", ascending=False).drop(columns=["abs_coefficient"])

# -----------------------------------------
# 4) Save outputs for report use
# -----------------------------------------
os.makedirs("reports/explainability", exist_ok=True)

results.to_csv("reports/explainability/logreg_feature_effects_full.csv", index=False)

# Save top 15 for easy report table
top15 = results.head(15)
top15.to_csv("reports/explainability/logreg_feature_effects_top15.csv", index=False)

# -----------------------------------------
# 5) Print a clean summary
# -----------------------------------------
print("Logistic Regression Explainability Output")
print("----------------------------------------")
print("Saved:")
print("- reports/explainability/logreg_feature_effects_full.csv")
print("- reports/explainability/logreg_feature_effects_top15.csv")

print("\nTop 10 Drivers (by absolute coefficient):")
print(top15.head(10).to_string(index=False))

print("\nInterpretation note:")
print("- Positive coefficient / odds_ratio > 1 => higher churn probability")
print("- Negative coefficient / odds_ratio < 1 => lower churn probability")