# eval_classifier3.py
import numpy as np
from LogisticRegression import LogisticRegression

# Load saved model and test data
data = np.load("classifier3_model.npz")
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
print(f"Classifier 3 (All features) Accuracy: {accuracy:.2f}%")
