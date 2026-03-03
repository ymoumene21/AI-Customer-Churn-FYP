import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# 1) Load dataset
df = pd.read_csv("data/processed/telco_features.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 2) Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4) Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 5) ROC + AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# 6) Save figure
os.makedirs("reports/figures", exist_ok=True)

plt.figure()
plt.plot(fpr, tpr, label=f"LogReg (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()

out_path = "reports/figures/roc_curve_logreg.png"
plt.savefig(out_path, dpi=200)
plt.close()

print("Saved ROC curve to:", out_path)
print("ROC-AUC:", round(auc, 4))