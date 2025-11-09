import pandas as pd
from pathlib import Path

# Build the path safely
csv_path = Path(__file__).resolve().parent.parent / "Phase1" / "data" / "IBM_Telco_Customer_Churn_raw.csv"
print("Loading dataset from:", csv_path)

# Read the CSV
df = pd.read_csv(csv_path)

# Basic info
print("\n✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nChurn distribution:")
print(df['Churn'].value_counts(normalize=True))
