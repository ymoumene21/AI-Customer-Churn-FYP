import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load cleaned data
df = pd.read_csv("data/cleaned/telco_cleaned.csv")

# Drop non-predictive ID
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify column types
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)

    ]
)

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Get feature names
encoded_cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
final_feature_names = list(numeric_cols) + list(encoded_cat_cols)

# Build final dataframe
X_final = pd.DataFrame(X_processed, columns=final_feature_names)
final_df = pd.concat([X_final, y.reset_index(drop=True)], axis=1)

# Save processed data
final_df.to_csv("data/processed/telco_features.csv", index=False)

print("✅ telco_features.csv created successfully")
