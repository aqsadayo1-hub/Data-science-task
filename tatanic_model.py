# TASK 1 : Titanic Survival Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------- STEP 1: Load dataset ---------------------
df = pd.read_csv("train.csv")  # Agar file folder ke andar hai to simple name chalega

# --------------------- STEP 2: Data Cleaning ---------------------
df = df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

# --------------------- STEP 3: Feature Selection ---------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------- STEP 4: Model Training ---------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --------------------- STEP 5: Model Evaluation ---------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
