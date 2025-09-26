
# eval_regression2_best_alt.py
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

iris = datasets.load_iris()

X = iris.data[:, [0, 2]]   # sepal length, petal length
y = iris.data[:, [3]]      # petal width

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, stratify=iris.target, random_state=42
)

# Evaluate no reg
model_no_reg = LinearRegression()
data_no_reg = np.load("regression2_alt_no_reg.npz")
model_no_reg.weights = data_no_reg["weights"]
model_no_reg.bias = data_no_reg["bias"]
mse_no_reg = model_no_reg.score(X_test, y_test)

# Evaluate with reg
model_reg = LinearRegression()
data_reg = np.load("regression2_alt_with_reg.npz")
model_reg.weights = data_reg["weights"]
model_reg.bias = data_reg["bias"]
mse_reg = model_reg.score(X_test, y_test)

print("Alt 2-Input Model (sepal length, petal length -> petal width)")
print(f"MSE without regularization: {mse_no_reg:.4f}")
print(f"MSE with regularization:    {mse_reg:.4f}")

