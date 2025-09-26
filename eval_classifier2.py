# eval_classifier2.py
import numpy as np
from LogisticRegression import LogisticRegression

# Load saved model and test data from training
data = np.load("classifier2_model.npz")
weights = data["weights"]
bias = data["bias"]
targets = data["targets"]
X_test = data["X_test"]
y_test = data["y_test"]

# Recreate classifier and load saved parameters
clf = LogisticRegression()
clf.weights = weights
clf.bias = bias
clf.targets = targets

# Evaluate accuracy on the test set
accuracy = clf.Accuracy(X_test, y_test)
print(f"Classifier 2 (Sepal features) Accuracy: {accuracy:.2f}%")
