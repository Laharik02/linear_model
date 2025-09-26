import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 1, 2]]   # sepal length, sepal width, petal length
y = iris.data[:, [3]]         # petal width

# Train/test split (same as training)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, stratify=iris.target, random_state=42
)

# Load trained model
model = LinearRegression()
data = np.load("regression1_no_reg.npz")
model.weights = data["weights"]
model.bias = data["bias"]

# Evaluate MSE
mse = model.score(X_test, y_test)
print("Model 1 (sepal length, sepal width, petal length -> petal width)")
print(f"MSE on test set: {mse:.4f}")
