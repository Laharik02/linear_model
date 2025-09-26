
### Linear Regression

* **Objective**: Predict continuous values using a linear relationship between input features and output targets.
* **Key Components**:

  * Implemented gradient descent to minimize **Mean Squared Error (MSE)**:
  * Added **early stopping**: training stops if validation loss fails to improve for a fixed patience.
  * Implemented **L2 Regularization (Ridge Regression)** to reduce overfitting by penalizing large weight values:
  * Extended the model to **multi-output regression**, predicting more than one target at once (e.g., predicting both petal length and petal width from sepal features).
    
* **Experiments**:
  * Trained four different regression models with different input/output feature combinations from the Iris dataset.
  * Compared models with and without regularization to observe the effect on weights and generalization.
  * Evaluated each model using **MSE** and **RMSE** on a held-out test set.

---

### Logistic Regression

* **Objective**: Perform **classification** on categorical data, specifically predicting the class of Iris flowers (Setosa, Versicolor, Virginica).
* **Key Components**:

  * Implemented **Softmax Regression** for multi-class classification:

  * Used **Cross-Entropy Loss** to measure classification error:

  * Trained with gradient descent, updating weights and bias iteratively.
  * Implemented accuracy calculation to measure performance on unseen test data.
* **Experiments**:

  * Trained classifiers using all four features of the Iris dataset.
  * Achieved **100% accuracy**, since the dataset is linearly separable.
  * Visualized decision regions to show clear separation between classes.

---

###  Learning Outcomes

By completing this assignment, the following concepts were demonstrated:

* Understanding how **linear regression** models map input features to continuous outputs.
* Applying **gradient descent** and early stopping for optimization.
* Using **regularization** to avoid overfitting.
* Extending regression to **multi-output predictions**.
* Implementing **logistic regression** with softmax for classification tasks.
* Comparing regression (continuous outputs) vs. classification (discrete outputs).
* Evaluating models with appropriate metrics: **MSE, RMSE, Accuracy**.



