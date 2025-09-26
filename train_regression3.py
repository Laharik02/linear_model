import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

# Load iris dataset
iris = datasets.load_iris()

# Features: sepal length, petal length, petal width
# Target: sepal width
X = iris.data[:, [0, 2, 3]]
y = iris.data[:, [1]]

# Train/test split (10% test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=iris.target, random_state=42
)

# Train model (no regularization)
model = LinearRegression(batch_size=32, max_epochs=100, patience=3, learning_rate=0.001)
model.fit(X_train, y_train)
model.save("regression3")



print("Weights without regularization (Model 3):\n", model.weights)


# Plot training loss
plt.plot(model.loss_per_cycle, label="No Regularization")
plt.legend()
plt.title("Training Loss (Model 3)")
plt.xlabel("Epochs(Iterations)")
plt.ylabel("MSE( Average Loss)")
plt.savefig("regression3_loss.png")
print("Model 3 trained and saved. Loss plot saved as regression3_loss.png")
plt.show()