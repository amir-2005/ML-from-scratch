import matplotlib.pyplot as plt
from MLP import MLP
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.datasets import make_moons
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from tabulate import tabulate
# -------------------------
# Dataset
# -------------------------
x, y = make_moons(n_samples=1200, random_state=8, noise=0.2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=2
)

# -------------------------
# MODEL 1 — Implemented MLP
# -------------------------
model1 = MLP(
    (5, 3),
    activation='tanh',
    learning_rate=0.1,
    epochs=2000,
    tol=1e-7,
    regularization_rate=1e-5
)

model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)

# -------------------------
# MODEL 2 — sklearn MLPClassifier
# -------------------------
model2 = MLPClassifier(
    hidden_layer_sizes=(5, 3),
    activation='tanh',
    solver='sgd',
    learning_rate_init=0.1,
    max_iter=2000,
    alpha=1e-5
)

model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)

# -------------------------
# Comparison Table
# -------------------------
results = {
    "Model": [" MLP", "sklearn MLP"],
    "Accuracy": [
        accuracy_score(y_test, y_pred1),
        accuracy_score(y_test, y_pred2)
    ],
    "Precision": [
        precision_score(y_test, y_pred1),
        precision_score(y_test, y_pred2)
    ],
    "Recall": [
        recall_score(y_test, y_pred1),
        recall_score(y_test, y_pred2)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred1),
        f1_score(y_test, y_pred2)
    ]
}

print()
print(tabulate(results, headers="keys"))
print()

# -------------------------
# Classification Reports
# -------------------------
print("\nImplemented MLP Report:")
print(classification_report(y_test, y_pred1))

print("\nsklearn MLP Report:")
print(classification_report(y_test, y_pred2))

# -------------------------
# Decision Boundaries
# -------------------------
plt.figure(figsize=(12, 5))

# Implemented MLP
plt.subplot(1, 2, 1)
plot_decision_regions(x, y, model1)
plt.title("Implemented MLP Decision Boundary")

# sklearn MLP
plt.subplot(1, 2, 2)
plot_decision_regions(x, y, model2)
plt.title("sklearn MLP Decision Boundary")

plt.tight_layout()
plt.show()
