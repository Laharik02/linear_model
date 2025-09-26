import numpy as np
from LogisticRegression import LogisticRegression
# Load saved model
data = np.load("classifier1_model.npz")

clf = LogisticRegression()
clf.weights = data["weights"]   # Match class attribute name
clf.bias = data["bias"]
clf.targets = int(data["targets"])

X_test = data["X_test"]
y_test = data["y_test"]

# Evaluate accuracy
acc = clf.Accuracy(X_test, y_test)
print(f"Classifier 1 (Petal features) Accuracy: {acc:.2f}%")

