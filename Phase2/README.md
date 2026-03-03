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


## Day 7 – Model Evaluation and Threshold Analysis

Day 7 focused on analysing the baseline Logistic Regression model in more depth to understand its prediction behaviour and real-world implications. A confusion matrix was used to examine true positives, true negatives, false positives, and false negatives, highlighting the types of errors made by the model.

Particular attention was given to false negatives, as these represent customers who churn but are not identified by the model. Precision and recall were analysed to better understand this trade-off, showing that recall is a more important metric for churn prediction than accuracy alone.

Threshold tuning was then performed by evaluating the model at different probability thresholds. Lower thresholds improved recall by identifying more at-risk customers, while higher thresholds increased precision but missed a larger proportion of churners. This demonstrated that the default threshold of 0.5 is not always optimal.

Overall, this analysis linked model evaluation metrics to business impact and provided a stronger justification for metric selection and decision-making in later model comparisons.


## Day 8 – Random Forest Model

Day 8 focused on implementing a Random Forest classifier as the first advanced model. The model was trained using the same train–test split as the baseline Logistic Regression to ensure a fair comparison.

Performance was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. The Random Forest model demonstrated improved ability to capture non-linear relationships in the data and provided a useful comparison point against the baseline model, particularly in terms of recall and overall discrimination performance.

### Model Comparison Insight

Although Random Forest was introduced as a more complex, non-linear model, it did not outperform the baseline Logistic Regression. In particular, recall and ROC-AUC were lower, indicating that the Random Forest model missed more churners and had weaker overall class discrimination.

Given the business importance of identifying at-risk customers, Logistic Regression remains the preferred model at this stage. This result highlights that increased model complexity does not necessarily lead to better performance and reinforces the importance of metric-driven model selection.


## Day 9 – XGBoost Model and Final Model Selection

Day 9 focused on implementing XGBoost as the final candidate model and comparing its performance against the baseline Logistic Regression and Random Forest models. XGBoost was selected due to its strong performance on tabular datasets and ability to model complex relationships.

The model was evaluated using the same metrics as previous models to ensure a fair comparison. While XGBoost outperformed Random Forest, it did not exceed Logistic Regression in terms of recall or ROC-AUC. As recall was prioritised for churn prediction, Logistic Regression remained the strongest-performing model.

This comparison demonstrated that increased model complexity does not necessarily lead to improved performance. Based on the evaluation results, Logistic Regression was selected as the final model due to its superior recall, strong discrimination ability, and interpretability.

## Explainability – Logistic Regression Feature Effects

An explainability step was added for the final selected model (Logistic Regression). Model coefficients were extracted and converted into odds ratios to identify the strongest drivers of churn and the direction of their effect. Outputs were exported as CSV tables to support interpretation in the report and presentation.

Explainability Section (Report)
4.X Model Explainability – Logistic Regression

To enhance interpretability, feature coefficients from the final Logistic Regression model were analysed and converted into odds ratios. This allowed identification of the strongest drivers of churn and the direction of their influence.

The analysis revealed that long-term contracts significantly reduce churn probability. Customers on two-year contracts were approximately 75% less likely to churn compared to the reference group. Similarly, increased tenure strongly reduced churn likelihood, indicating that customer loyalty increases over time.

Conversely, customers using fibre optic internet services exhibited a significantly higher probability of churn, with an odds ratio of approximately 3.0. Higher total charges and the use of electronic check payment methods were also associated with increased churn risk.

These findings provide business-relevant insights, demonstrating that contract structure, pricing, and service type are key drivers of customer retention. This explainability analysis strengthens the validity of the selected model and improves its practical applicability.


Cross-Validation and Model Robustness

To ensure that model performance was not dependent on a single train–test split, 5-fold Stratified Cross-Validation was performed on the selected Logistic Regression model. Results showed stable performance across folds, with low standard deviation in recall and accuracy. This confirmed that the model generalises well and is not highly sensitive to data partitioning.

Visual Evaluation – ROC Curve and Confusion Matrix

To enhance evaluation clarity, a ROC curve was generated to visualise class discrimination performance, and a confusion matrix heatmap was created to illustrate prediction outcomes. These visual tools provide an intuitive understanding of the model’s trade-offs between false positives and false negatives.

Class Imbalance Handling Experiment

An additional Logistic Regression model using class_weight='balanced' was tested to evaluate alternative imbalance handling strategies. While recall significantly improved, precision and overall accuracy decreased. The baseline model was retained due to its more balanced performance profile.

Regularisation Sensitivity Analysis

A sensitivity analysis was conducted by testing multiple regularisation strengths (C values). Performance metrics remained stable across configurations, indicating that the model is robust and not highly sensitive to regularisation tuning.

Business-Oriented Threshold Selection

A cost-based threshold analysis was performed to align model evaluation with business objectives. By assigning higher cost to false negatives, a lower decision threshold was shown to reduce overall estimated cost. This demonstrates that the default 0.5 threshold is not always optimal in a churn prediction context.