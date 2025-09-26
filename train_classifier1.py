import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
# Fix randomness for reproducibility
np.random.seed(42)
# Load dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]   # Petal length, Petal width
y = iris.target

# Train/test split (10% test size)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Train classifier
clf = LogisticRegression(learning_rate=0.01, epochs=1000, L2_term=0.01)
clf.fit(X_train, y_train)

# Save model
np.savez("classifier1_model.npz",
         weights=clf.weights,
         bias=clf.bias,
         targets=clf.targets,
         X_test=X_test,
         y_test=y_test)


# Accuracy
acc = clf.Accuracy(X_test, y_test)
print(f"Classifier 1 (Petal features) Accuracy: {acc:.2f}%")

# Visualization (2D only)
plt.figure(figsize=(8,6))
plot_decision_regions(X=X, y=y, clf=clf, legend=2)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Logistic Regression Decision Regions (Petal Features)")
plt.legend(["Setosa", "Versicolor", "Virginica"])
plt.savefig("classifier1_decision_regions.png", dpi=300)  # saves plot as PNG
plt.show()

