📘 Task 1: Titanic Survival Classification – ReadMe File
This task focuses on building a machine learning model to predict the survival of passengers from the Titanic disaster. Using the Titanic dataset, the goal is to analyze important features such as age, gender, ticket class, fare, and family size to determine their impact on survival. The project involves data cleaning, preprocessing, feature engineering, model training, and evaluation. By applying machine learning techniques, we aim to extract meaningful insights and create an accurate classification model that can predict whether a passenger survived or not.

2. Dataset Description

Dataset: train.csv (Kaggle Titanic dataset)

Target variable: Survived

Features:

Pclass

Sex

Age

SibSp

Parch

Fare

Embarked

3. Steps Performed
Step 1: Data Loading

Pandas ka use kar ke train.csv load kiya.

Step 2: Data Cleaning & Preprocessing

Missing values fill kiye

Categorical columns encode kiye (Sex, Embarked)

Irrelevant columns drop kiye (Name, Ticket, Cabin)

Step 3: Feature Selection

Final features:

Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

Step 4: Train-Test Split

80% training
20% testing

Step 5: Model Training

RandomForestClassifier use kiya.

Step 6: Model Evaluation

Accuracy calculate ki

Confusion Matrix

Classification Report


4. Model Output

Example output (Aqsa ka actual output):

Accuracy: 0.81

Confusion Matrix:
[[88 17]
 [17 57]]

Classification Report:
              precision   recall   f1-score   support
0               0.84      0.84      0.84      105
1               0.77      0.77      0.77       74


5. Conclusion

Random Forest model ne 81% accuracy achieve ki jo kaafi acha performance hai. Model ne majority classes ko theek predict kiya, aur future improvement ke liye hyperparameter tuning ya advanced models use kiye ja sakte hain.







