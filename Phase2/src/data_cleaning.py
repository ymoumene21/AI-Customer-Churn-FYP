import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/telco_raw.csv")

# Drop duplicates
df = df.drop_duplicates()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv("data/cleaned/telco_cleaned.csv", index=False)

print("✅ telco_cleaned.csv created successfully")
