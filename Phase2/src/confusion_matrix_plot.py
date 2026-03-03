import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 1) Load dataset
df = pd.read_csv("data/processed/telco_features.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 2) Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4) Predictions
y_pred = model.predict(X_test)

# 5) Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 6) Save heatmap
os.makedirs("reports/figures", exist_ok=True)

plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix — Logistic Regression")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["No Churn", "Churn"])
plt.yticks(tick_marks, ["No Churn", "Churn"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center")

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()

out_path = "reports/figures/confusion_matrix_logreg.png"
plt.savefig(out_path, dpi=200)
plt.close()

print("Saved confusion matrix to:", out_path)