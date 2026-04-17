Titanic Survival Prediction: End-to-End ML Pipeline

Advanced Data Analytics & Model Calibration

This project is an approach to the Titanic survival classification problem. It goes beyond simple model fitting to implement production-grade pipelines, imbalanced class handling (SMOTE), and probability calibration.
 Key Features

    Automated Data Preprocessing: Integrated ColumnTransformer for handling numerical (scaling) and categorical (OneHotEncoding) data within a unified pipeline.

    Imbalance Management: Utilized SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in survival rates, ensuring the model identifies minority class patterns effectively.

    Model Benchmarking: Evaluated multiple architectures including Logistic Regression, SVM, Random Forest, and XGBoost.

    Probability Calibration: Implemented Brier Score analysis and Calibration Curves to refine prediction confidence levels.

    Feature Engineering: Engineered custom features like FamilySize and utilized domain-specific imputation (Age by Pclass median).

 Technical Stack

    Language: Python

    Libraries: Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, XGBoost

    Visualization: Matplotlib, Seaborn

    Deployment-Ready: Models serialized via joblib for integration into REST APIs (FastAPI).

 Methodology & Analysis
1. Data Audit & Imputation

    Age: Imputed using the median age of the passenger's specific class (Pclass), preserving class-based demographic nuances.

    Dropped Features: Removed identifiers with no predictive power (Name, Ticket, PassengerId) and features with >75% missing data (Cabin).

2. The Pipeline
   
   Python

# Example of the robust pipeline used in this project
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=5))
])

3. Model Evaluation

The models were evaluated using a suite of metrics beyond simple accuracy:

    ROC-AUC Score: To measure the model's ability to distinguish between classes.

    Brier Score: Used specifically during calibration to measure the accuracy of predicted probabilities.

 Key Findings

    Feature Importance: Sex, Pclass, and Fare were identified as the primary drivers of survival, which aligns with historical "women and children first" protocols.

    Calibration: The calibrated XGBoost model showed the most reliable alignment between predicted probability and actual survival frequency.
