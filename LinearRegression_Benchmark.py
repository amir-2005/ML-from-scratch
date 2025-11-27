import numpy as np
from LinearRegression import LinReg
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

# ---------- DATA ----------
data = make_regression(n_samples=333, n_features=3, noise=0.33, coef=True, shuffle=True)
x, y = data[0], data[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# ---------- MODEL 0 (sklearn) ----------
model0 = LinearRegression()
model0.fit(x_train, y_train)
y_pred0 = model0.predict(x_test)

# ---------- MODEL 1 (Implemented GD + Lasso model) ----------
model1 = LinReg(method="GD", reg_method="Lasso", learning_rate=1e-5, n_iter=1000)
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)

# ---------- MODEL 2 (Normal Equation) ----------
model2 = LinReg(method="NE")
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)

# ---------- Metrics ----------
results = {
    "Model": ["Model0 (sklearn)", "Model1 (GD + Lasso)", "Model2 (Normal Eq)"],
    "RMSE": [
        root_mean_squared_error(y_test, y_pred0),
        root_mean_squared_error(y_test, y_pred1),
        root_mean_squared_error(y_test, y_pred2),
    
    ],
    "R2 Score": [
        r2_score(y_test, y_pred0),
        r2_score(y_test, y_pred1),
        r2_score(y_test, y_pred2),
    ],
    "Estimated Coefs": [
        np.array([model0.intercept_, *model0.coef_]),
        model1._coef,
        model2._coef
    ]
}

print()
print(tabulate(results, headers="keys"))
print("\nReal coefficients:", data[2])
