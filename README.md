# Transaction-Fraud-Detector

## Problem Statement:
The rapid growth of digital transactions has led to a significant increase in online payment fraud, posing a major threat to financial institutions and their customers. Traditional rule-based fraud detection systems are often slow, rigid, and unable to adapt to the evolving tactics of fraudsters. This results in substantial financial losses and a decline in customer trust. To address this challenge, a robust, data-driven solution is required. This project aims to develop a machine learning model that can accurately and efficiently detect fraudulent online payment transactions in real-time, thereby minimizing financial losses and enhancing the security of the digital payment ecosystem.

## Dataset:
The analysis is based on a historical dataset of online transactions. This dataset contains records of both legitimate and fraudulent activities, providing the necessary information to train and evaluate predictive models.
step --------------- tells about the unit of time
type --------------- type of transaction done
amount ------------ the total amount of transaction
nameOrg	----------- account that starts the transaction 
oldbalanceOrg ----- Balance of the account of sender before transaction
newbalanceOrg ----- Balance of the account of sender after transaction
nameDest ---------- account that receives the transaction
oldbalanceDest ---- Balance of the account of receiver before transaction
newbalanceDest ---- Balance of the account of receiver after transaction
isFraud	----------- The value to be predicted i.e. 0 or 1

## Workflow
### Data Overview
* Loaded the dataset and performed an initial inspection to understand its structure, size, and data types.
* Checked for and confirmed the absence of missing values and duplicate records, ensuring data quality.

### Data Preprocessing & Feature Engineering
* Dropped irrelevant identifier columns (nameOrig, nameDest) that do not contribute to predictive power.
* Engineered new time-based features from the step column to create day, hour_of_day, and time_segment (e.g., Morning, Afternoon), adding valuable temporal context.

### Applied feature encoding techniques
* Label Encoding: Converted the categorical time_segment feature into numerical values.
* One-Hot Encoding: Transformed the type of transaction column into separate binary columns to make it suitable for modeling.

### Model Development
* Split the data into training (70%) and testing (30%) sets, using stratification to maintain the same proportion of fraudulent transactions in both splits.
* Developed a suite of classification models, including Logistic Regression, Naive Bayes, K-Nearest Neighbors, Decision Tree, and advanced ensemble methods like Random Forest, AdaBoost, Gradient Boosting, LightGBM, and XGBoost.
* Systematically evaluated model performance using key metrics such as Accuracy, Precision, Recall, F1-Score, Cohen Kappa, and ROC AUC score.
* Utilized GridSearchCV for hyperparameter tuning on the more complex models to optimize their predictive performance.

## Final Model Selection
* Compared all trained models based on their evaluation metrics, with a focus on Cohen Kappa and ROC AUC as strong indicators of performance on an imbalanced dataset.
* Based on the results, the XGBoost Classifier demonstrated superior performance, achieving an outstanding Cohen’s Kappa score of 0.898 and an excellent ROC AUC score of 0.995, indicating high accuracy and strong predictive capability for the target variable.
* Validated the final model's stability and robustness using K-Fold Cross-Validation, which confirmed its high performance with an average ROC AUC score of 0.996.

## Deployment Preparation
* The trained and validated XGBoost model was saved into a final_model.joblib file for deployment in a production environment.

## Key Features Analyzed
- Transaction Type: Categorical feature indicating the nature of the transaction (e.g., CASH_OUT, TRANSFER).
- Transaction Amount: The monetary value of the transaction.
- Account Balances: Balances of both the origin and destination accounts before and after the transaction (oldbalanceOrg, newbalanceOrig, etc.).
- Engineered Time Features: day and time_segment were created to capture temporal patterns in fraudulent activities.

## Tools & Technologies
* Python (pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels)
* Jupyter Notebook

## Future Work
* Deploy the saved model as a real-time API to score incoming transactions and flag potential fraud instantly.
* Develop a monitoring dashboard to track the model's performance in production and identify concept drift.
* Explore more advanced techniques like anomaly detection or graph-based neural networks to capture complex fraud patterns.
* Implement a retraining pipeline to periodically update the model with new transaction data, ensuring it remains effective against new fraud tactics.
