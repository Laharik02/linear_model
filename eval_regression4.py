import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

iris = datasets.load_iris()
X = iris.data[:, [1, 2, 3]]   # sepal width, petal length, petal width
y = iris.data[:, [0]]         # sepal length

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, stratify=iris.target, random_state=42
)

model = LinearRegression()
data = np.load("regression4.npz")
model.weights = data["weights"]
model.bias = data["bias"]

mse = model.score(X_test, y_test)
print("Model 4 (sepal length, petal length, petal width -> sepal width)")

print(f"MSE on test set: {mse:.4f}")