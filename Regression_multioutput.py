import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression  # your implemented class

# Load iris dataset
iris = datasets.load_iris()

# Inputs: Sepal length (0), Sepal width (1)
X = iris.data[:, [0, 1]]

# Outputs: Petal length (2), Petal width (3)
y = iris.data[:, [2, 3]]

# Train/test split (10% test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=iris.target, random_state=42
)

# Initialize model
model_multiple = LinearRegression(batch_size=16, max_epochs=100, patience=3, learning_rate=0.01)

# Train the model
model_multiple.fit(X_train, y_train)

# Save trained parameters
model_multiple.save("multioutput_regression")

# Predict on test data
y_pred = model_multiple.predict(X_test)

# Evaluate Mean Squared Error for multiple outputs
mse = model_multiple.score(X_test, y_test)

print("Predicted values (first 5):\n", y_pred[:5])
print("True values (first 5):\n", y_test[:5])
print("Mean Squared Error on Test Set =", mse)

# Plot training loss
plt.plot(model_multiple.loss_per_cycle, label="Training Loss")
plt.legend()
plt.title("Multi-Output Regression Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss (average per batch)")
plt.savefig("multioutput_regression_loss.png")
plt.show()
