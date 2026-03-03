# day4_feature_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Paths ----------
# Resolve repo root and dataset path safely (handles spaces in path)
ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / "Phase1" / "data" / "IBM_Telco_Customer_Churn_raw.csv"

# ---------- Load ----------
df = pd.read_csv(data_path)

# ---------- Normalize column names (optional but safer) ----------
# Keep original names but prepare a case-insensitive lookup
cols_lower = {c.lower(): c for c in df.columns}

def get_col(name: str):
    """Return actual column name by case-insensitive key, else None."""
    return cols_lower.get(name.lower())

# ---------- Clean / Cast ----------
# TotalCharges sometimes has blanks in the IBM dataset; coerce to numeric
tc_col = get_col("TotalCharges")
if tc_col:
    df[tc_col] = pd.to_numeric(df[tc_col], errors='coerce')

# Make a numeric Churn target if present
churn_col = get_col("Churn")
churn_bin = None
if churn_col:
    # Map Yes/No (and be resilient to lowercase/whitespace)
    df[churn_col] = df[churn_col].astype(str).str.strip().str.title()
    mapping = {"Yes": 1, "No": 0}
    if set(df[churn_col].unique()) <= set(mapping.keys()):
        df["__churn_bin__"] = df[churn_col].map(mapping)
        churn_bin = "__churn_bin__"

# ---------- Basic summaries ----------
summary = df.describe(include='all').transpose()
missing = df.isna().sum().sort_values(ascending=False)

# ---------- Numeric correlations with churn ----------
corr_with_churn = None
if churn_bin:
    num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove the churn column itself from the feature list if present
    num_feats = [c for c in num_feats if c != churn_bin]
    # Compute Pearson correlation of each numeric feature with churn_bin
    corr_with_churn = (
        df[num_feats + [churn_bin]]
        .corr(numeric_only=True)[churn_bin]
        .drop(churn_bin)
        .abs()  # absolute strength
        .sort_values(ascending=False)
        .to_frame(name="abs_corr_with_churn")
    )

# ---------- Save reports ----------
out_dir = ROOT / "reports"
out_dir.mkdir(exist_ok=True, parents=True)

summary.to_csv(out_dir / "day4_summary.csv")
missing.to_csv(out_dir / "day4_missing.csv")
if corr_with_churn is not None:
    corr_with_churn.to_csv(out_dir / "day4_corr_numeric_vs_churn.csv")

print("✅ Saved:")
print(f" - {out_dir / 'day4_summary.csv'}")
print(f" - {out_dir / 'day4_missing.csv'}")
if corr_with_churn is not None:
    print(f" - {out_dir / 'day4_corr_numeric_vs_churn.csv'}")
else:
    print(" - Skipped correlation: 'Churn' not found or not mappable to Yes/No.")
