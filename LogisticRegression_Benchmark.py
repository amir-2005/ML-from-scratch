import pandas as pd
from LogisticRegression import LogisticReg
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

# ------------------------------
# Load dataset
# ------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# MODEL 0 — sklearn Logistic Regression
# ------------------------------
model0 = LogisticRegression(max_iter=5000)
model0.fit(X_train, y_train)
y_pred0 = model0.predict(X_test)

# ------------------------------
# MODEL 1 — Implemented Logistic Regression
# ------------------------------
model1 = LogisticReg(
    learning_rate=0.01,
    n_iter=5000,
    reg_rate=0 # No regularization
)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

# ------------------------------
# Build comparison table
# ------------------------------
results = {
    "Model": ["Model0 (sklearn)", "Model1 (Custom GD)"],
    "Accuracy": [
        accuracy_score(y_test, y_pred0),
        accuracy_score(y_test, y_pred1),
    ],
    "Precision": [
        precision_score(y_test, y_pred0),
        precision_score(y_test, y_pred1),
    ],
    "Recall": [
        recall_score(y_test, y_pred0),
        recall_score(y_test, y_pred1),
    ],
    "F1 Score": [
        f1_score(y_test, y_pred0),
        f1_score(y_test, y_pred1),
    ]
}

print(tabulate(results, headers="keys"))
