# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and import required Python libraries for Decision Tree classification.

2.Preprocess the data by converting categorical values into numerical form and split the dataset into training and testing sets.

3.Train the Decision Tree Classifier model using the training data.

4.Test and evaluate the model using testing data and calculate Accuracy, Confusion Matrix, and Classification Report.

## Program:
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SAADHANA A
RegisterNumber: 25018432 
*/
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\acer\Downloads\Employee.csv")   # change file name if needed
print("Dataset Loaded Successfully\n")
print(data.head(), "\n")

data = pd.get_dummies(data, drop_first=True)
y = data['status_Placed'] if 'status_Placed' in data.columns else data['left']
X = data.drop(y.name, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred), "\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), "\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Output:

<img width="451" height="481" alt="Screenshot 2026-02-21 203440" src="https://github.com/user-attachments/assets/f344b33d-6dbc-49cb-924d-8c05916c7440" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
