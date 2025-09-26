import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

# Fix randomness for reproducibility
np.random.seed(42)

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data   # all 4 features
y = iris.target

# Train/test split (10% test set, stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True, random_state=42, stratify=y
)

# Train classifier (tuned for high accuracy)
clf = LogisticRegression(
    learning_rate=0.001,   # smaller LR for stable convergence
    epochs=5000,           # more iterations to fully converge
    L2_term=0.0            # no regularization (Iris is clean & separable)
)
clf.fit(X_train, y_train)

# Save trained model + test set
np.savez("classifier3_model.npz",
         weights=clf.weights,
         bias=clf.bias.reshape(1, -1),  # ensure consistent shape
         targets=clf.targets,
         X_test=X_test,
         y_test=y_test)

# Accuracy on test set
acc = clf.Accuracy(X_test, y_test)
print(f"Classifier 3 (All features) Accuracy: {acc:.2f}%")
