import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Load dataset
df = pd.read_csv("data/processed/telco_features.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Predictions at default threshold (0.5)
y_pred_default = (y_prob >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_default)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix (Threshold = 0.5)")
print(cm)
print("\nTN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

print("\nPrecision:", precision_score(y_test, y_pred_default))
print("Recall   :", recall_score(y_test, y_pred_default))

print("\nThreshold Tuning Results\n")

for threshold in [0.3, 0.4, 0.5, 0.6]:
    y_pred_thresh = (y_prob >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    print(f"Threshold {threshold:.1f} -> Precision: {precision:.3f}, Recall: {recall:.3f}")
