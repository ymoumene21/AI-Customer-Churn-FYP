## Day 5 – Feature Engineering and Dataset Preparation

### Tasks Completed
- Loaded the cleaned Telco Customer Churn dataset
- Removed non-predictive identifier columns
- Separated input features and target variable
- Identified numerical and categorical features
- Applied feature scaling to numerical variables
- Applied one-hot encoding to categorical variables
- Reconstructed feature names for interpretability
- Created a final model-ready dataset

### Output
- `data/processed/telco_features.csv`

### Outcome
By the end of Day 5, the dataset was fully transformed into a numerical, machine-learning-ready format. All features were consistently scaled and encoded, enabling reliable baseline and advanced model training in later phases.


## Day 6 – Baseline Logistic Regression

### Tasks Completed
- Loaded final processed dataset
- Split data into training and testing sets (80/20)
- Trained a baseline Logistic Regression model
- Evaluated performance using accuracy, precision, recall, F1-score, and ROC-AUC

### Outcome
The baseline model provides a reference point for evaluating more complex models in later phases. Recall and ROC-AUC were prioritised due to the imbalanced nature of the churn dataset.
