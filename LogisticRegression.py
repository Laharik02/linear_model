import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from mlxtend.plotting import plot_decision_regions
np.random.seed(42)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=2000, L2_term=0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.L2_term = L2_term
        self.weights = None
        self.bias = None
        self.targets = None
        self.losses = []

    def softmax(self, z):
        # stable softmax (avoids overflow)
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def hot_encoding(self, y):
        return np.eye(self.targets)[y]

    def Cross_entropy_loss(self, y_pred, One_hot_y):
        loss = -np.sum(One_hot_y * np.log(y_pred + 1e-9)) / y_pred.shape[0]
        reg = (self.L2_term / 2) * np.sum(np.square(self.weights))
        return loss + reg

    def gradients(self, X, y_pred, One_hot_y):
        n = X.shape[0]
        dw = np.dot(X.T, (y_pred - One_hot_y) / y_pred.shape[0])
        db = np.mean(y_pred - One_hot_y, axis=0)
        return dw, db

    def fit(self, X, y, learning_rate=0.01, epochs=2000, L2_term=0):
        n_samples, n_features = X.shape
        Dimensions = X.shape[1]
        self.targets = len(np.unique(y))
        self.epochs = epochs
        self.L2_term = L2_term
        self.learning_rate = learning_rate

        # init weights & bias
        self.weights = np.random.randn(Dimensions, self.targets)
        self.bias = np.random.randn(self.targets)

        # one-hot encode targets
        One_hot_y  = self.hot_encoding(y)

        for i in range(self.epochs):
            # linear combination
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)

            # loss
            loss = self.Cross_entropy_loss(y_pred, One_hot_y)
            self.losses.append(loss)

            # gradients
            dw, db = self.gradients(X, y_pred, One_hot_y)

            # update
            self.weights -= self.learning_rate * (dw + self.L2_term * self.weights)
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                self.losses.append(loss)
        # Function fro one hot encoding the data.
 
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        probs = self.softmax(z)
        return np.argmax(probs, axis=1)
    
    def Accuracy(self, X_test, y_test):

        # Finding the predictions.
        y_pred = self.predict(X_test)
        # Comparing matching values between the predictions and the y_test.
        c_pred = sum([1 for i in range(len(y_test)) if y_test[i] == y_pred[i]])
        # Returns the percentage of the Accurate predictions
        return (c_pred/len(y_test))*100