import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import sys
print("Python executable in use:", sys.executable)
np.random.seed(42)
class LinearRegression:

  def __init__(self, batch_size=32, max_epochs=100, patience =3, learning_rate = .001):
      self.batch_size = batch_size
      self.max_epochs = max_epochs
      self.patience = patience
      self.learning_rate = learning_rate
      self.loss_per_cycle = []
      self.weights, self.bias = None, None
      self.weights_without_Regularization = None


  def fit(self, X, y, batch_size =32, max_epochs = 100, patience = 3, learning_rate = 0.01, regularization=0.0):
      self.batch_size = batch_size
      self.max_epochs = max_epochs
      self.patience = patience
      self.learning_rate = learning_rate
      self.regularization = regularization

      # training data 
    # #   y = y.reshape(-1, 1)
    #   if y.ndim == 1:
    #      y = y.reshape(-1,1)
      n_s, d_f, m_o = X.shape[0], X.shape[1], y.shape[1]


      # Initializing the weight and bias matrix parameters
      self.weights = np.random.randn(d_f, m_o) * 0.05
      self.bias = np.random.randn(1, m_o)
      
      batch_limit = int(0.9 * n_s)
      # Splitting the training data into train and validation set with 90 percent for training and 10% for validation
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)

      # Assigning the best parameters into variable for comapring it with the updated values while optimizing
      #early stopping
      best_weights, best_bias = self.weights, self.bias
      best_loss, patience_counter = float("inf"), 0

      # Gradient Descent
      #max number of times the model should train through the data set 
      for epoch in range(max_epochs):
        count = 0 # To track the no of loss that occurs within a single batch
        loss_per_batch = 0 # Initializing the loss to be zero so that after each iteration the values will be zero.
       
        #mini batch loop for gradient descent 
        for i in range(0, batch_limit, batch_size):

          # Selecting a mini-batch of data from the training set for use in the optimization process.
          # loss_per_batch = 0 # Initializing the loss to be zero so that after each iteration the values will be zero.
          X_batch = X_train[i:i + batch_size]
          y_batch = y_train[i:i + batch_size]

          # Prediction and finding the loss
          y_pred = X_batch.dot(self.weights) + self.bias
          loss = np.mean((y_batch-y_pred)** 2) #Mean squre error (MSE)

          loss_per_batch += loss # Adding up the loss within the batch
          count += 1 # To take average of the loss within one batch
          
          # calculating the gradients of the cost function with respect the weights and the bias.
          #gradients L2 regularization (adds penalty)
          dw = (2 / batch_size) * (X_batch.T).dot(y_pred - y_batch)  + regularization * self.weights 
          db = (2 / batch_size) * np.sum(y_pred - y_batch)

          # Updating weights and bias 
          self.weights -= learning_rate * dw
          self.bias -= learning_rate * db

        # Appending the averaged loss of each batch to a list for visualization purpose
        average_loss = loss_per_batch / count
        self.loss_per_cycle.append(average_loss)
      

        # Predicting values for the target values with the best values of the weights
        y_pred = X_val.dot(self.weights) + self.bias

        # Calculates the MSE of on the validation set values
        val_loss = np.mean((y_val - y_pred)** 2)



         # Comparing validation loss with the best loss so far
        if val_loss < best_loss:
          best_loss = val_loss
          best_weights, best_bias = self.weights.copy(), self.bias.copy()
          patience_counter = 0
        else:
          patience_counter +=1
          if patience_counter >= patience: # Checking if the patience count becomes equal to or more than the desired patience count training gets stopped.
            break
 
      # after training, restore the best parameters 
      self.weights = best_weights
      self.bias = best_bias
      self.weights_without_Regularization = best_weights # Saving best weights 

  def predict(self, X):
    y_pred = X.dot(self.weights) + self.bias
    return y_pred

  # The score method to determine the MSE of the model
  def score(self, X, y):
    y_pred = self.predict(X)
    MSE = np.mean((y-y_pred)**2)
    return MSE

  # Root Mean Square error Method
  def RMSE(self, y_predicted, y):
    return np.sqrt(np.mean((y_predicted - y) ** 2))
  
  
  def save(self, filepath):
    np.savez(filepath, weights=self.weights, bias=self.bias)
    print(f"Model parameters saved to {filepath}.npz")

  def load(self, filepath):
    data = np.load(filepath + ".npz")
    self.weights = data["weights"]
    self.bias = data["bias"]
    print(f"Model parameters loaded from {filepath}.npz")
  
class L2Regularization(LinearRegression):
    def __init__(self, batch_size=32, max_epochs=100, patience=3, learning_rate=0.001, regularization=0.001):
        super().__init__(batch_size, max_epochs, patience, learning_rate)
        self.regularization = regularization
        self.weights_with_Regularization = None
        self.loss_per_cycle_with_R = []

    def fit_with_R(self, X, y, batch_size=32, max_epochs=100, patience=3, learning_rate=0.001, regularization=0.001):
        # just call parent fit with regularization
        super().fit(X, y, batch_size, max_epochs, patience, learning_rate, regularization)
        self.regularization = regularization
        self.weights_with_Regularization = self.weights.copy()
        self.loss_per_cycle_with_R = self.loss_per_cycle.copy()






