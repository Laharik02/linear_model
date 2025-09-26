

# train_regression2_reg_2inputs.py
# train_regression2_best_alt.py
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

iris = datasets.load_iris()

# Inputs: sepal length, petal length -> Output: petal width
X = iris.data[:, [0, 2]]
y = iris.data[:, [3]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=iris.target, random_state=42
)

# No regularization
model_no_reg = LinearRegression(batch_size=32, max_epochs=1000, patience=5, learning_rate=0.001)
model_no_reg.fit(X_train, y_train)
model_no_reg.save("regression2_alt_no_reg")

# With regularization
model_reg = LinearRegression(batch_size=32, max_epochs=1000, patience=5, learning_rate=0.001)
model_reg.fit(X_train, y_train, regularization=0.001)
model_reg.save("regression2_alt_with_reg")

# Compare weights
print("Weights no reg:\n", model_no_reg.weights)
print("Weights with reg:\n", model_reg.weights)

# Plot loss
plt.plot(model_no_reg.loss_per_cycle, label="No Regularization")
plt.plot(model_reg.loss_per_cycle, label="With L2 Regularization")
plt.legend()
plt.title("Training Loss Comparison (Alt 2-Input Model)")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.savefig("regression2_alt_loss.png")
plt.show()

