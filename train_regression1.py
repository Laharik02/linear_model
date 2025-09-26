# train_regression1_no_reg.py
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

iris = datasets.load_iris()
X = iris.data[:, [0, 1, 2]]   # sepal length, sepal width, petal length
y = iris.data[:, [3]]         # petal width

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=iris.target, random_state=42
)

# Model without regularization
model_no_reg = LinearRegression(batch_size=32, max_epochs=100, patience=3, learning_rate=0.001)
model_no_reg.fit(X_train, y_train)
model_no_reg.save("regression1_no_reg")

print("Weights without regularization (Model 1):\n", model_no_reg.weights)

# Plot training loss
plt.plot(model_no_reg.loss_per_cycle, label="No Regularization")
plt.legend()
plt.title("Training Loss (Model 1 - No Regularization)")
plt.xlabel("Epochs(Iterations)")
plt.ylabel("MSE (Average Loss)")
plt.savefig("regression1_no_reg_loss.png")
plt.show()


#
# Why train_regression1 stops at ~18 epochs

# Early stopping kicks in sooner.

# In your fit loop, training halts when the validation loss hasn’t improved for patience steps.

# For train_regression1, the validation set happens to plateau around epoch 18. So early stopping triggers there.

# Different data split sensitivity.

# In train_regression1, the model maps (sepal length, sepal width, petal length) → petal width.

# This mapping is tighter and easier to learn (petal width is strongly correlated with petal length in the iris dataset).

# That means the model converges very quickly → validation loss doesn’t improve much after ~18 epochs → early stopping halts training.

# Other regressions are harder tasks.

# For example, train_regression2 predicts petal length from different features. That relationship is a little noisier.

# Because it’s harder, the validation loss keeps improving slowly → early stopping doesn’t trigger → training runs for all 100 epochs.

# Why “17.5 epochs” shows up on the plot

# As I explained before, you didn’t actually train for half an epoch — the x-axis ticks are just spaced by 2.5.
# So “17.5” really means you trained for 18 epochs before early stopping.

# ✅ So in short:

# train_regression1 stops earlier because the task converges quickly and validation loss plateaus.

# The other regression models run longer because their validation loss keeps improving (no early stop).

# The “.5” is just matplotlib’s auto tick labeling, not real fractional epochs.