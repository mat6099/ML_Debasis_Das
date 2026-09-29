# Chapter-3:  A Toure Of Machine Learning Classisfication Using Scikit-Learn

## Introduction

In Chapter 2 we wrote the Perceptron and Adaline ourselves using NumPy. In this chapter we start using **scikit-learn**, a Python library that gives ready-made, fast and easy-to-use ML algorithms. We also learn **logistic regression**, one of the most popular classification models, and **regularization**, which helps to stop overfitting.

## Summery

### 1. Choosing a Classification Algorithm
According to the **"No Free Lunch" theorem**, no single algorithm is best for every problem. So it is always good to try a few algorithms and compare them. Training a supervised model has 5 main steps:

1. Select features and collect labelled data
2. Choose a performance metric
3. Choose an algorithm and train the model
4. Evaluate the model
5. Tune the settings to improve the model

### 2. First Steps with Scikit-Learn — Training a Perceptron
- The **Iris dataset** is loaded directly from scikit-learn. Only two features are used: petal length and petal width.
- `train_test_split` splits the data into **70% training** and **30% test** data. `stratify=y` keeps the same class ratio in both parts, and `random_state` makes the results repeatable.
- `StandardScaler` standardizes the features. We learn the mean and standard deviation from the **training data only**, then apply the same values to the test data.
- The `Perceptron` class handles all 3 flower classes using **One-versus-Rest (OvR)**.
- The model makes only **1 mistake out of 45** test flowers, giving **97.8% accuracy** (`accuracy_score` or `.score()`).
- The `plot_decision_regions` function from Chapter 2 is updated to highlight the test samples.
- **Problem:** The three classes cannot be separated perfectly by straight lines, so the Perceptron never fully converges. For this reason it is not recommended in practice.

### 3. Logistic Regression — Predicting Probabilities
Despite its name, logistic regression is used for **classification**, not regression. It is simple, and it is one of the most widely used classifiers in industry.

- **Odds** = p / (1 − p), where p is the probability of the positive class
- **Logit** = log(odds). It converts a probability (0 to 1) into any real number.
- **Sigmoid function** `σ(z) = 1 / (1 + e^(−z))` does the reverse. It squeezes any number into the range 0 to 1, in an **S-shaped curve**.
- **Adaline vs logistic regression:** The only difference is the activation function. Adaline uses the identity function `σ(z) = z`, and logistic regression uses the sigmoid.
- **Prediction:** If σ(z) ≥ 0.5, predict class 1. Otherwise, predict class 0.
- **Why probabilities are useful:** Weather forecasting can report the *chance* of rain, and doctors can estimate the *chance* of a disease.

### 4. Learning the Weights — Logistic Loss
- The model tries to **maximize the likelihood** of the correct labels. Taking the log makes the maths easier and avoids very small numbers.
- Flipping the sign gives the **logistic loss (log loss)**, which we minimize with gradient descent:
  `L = −y·log(σ(z)) − (1 − y)·log(1 − σ(z))`
- **Simple meaning:** A correct prediction gives a loss near 0. A wrong prediction gives a **very large** loss, so the model is punished heavily for wrong answers.

### 5. Converting Adaline into Logistic Regression
- We take the `AdalineGD` code from Chapter 2 and change just two things: the **activation** (use the sigmoid) and the **loss** (use log loss). This gives the `LogisticRegressionGD` class.
- The weight update rule stays the same as in Adaline: `w = w + η·(y − ŷ)·x`
- Our own version works only for **binary** classification, so it is tested only on Setosa vs Versicolor.

### 6. Logistic Regression with Scikit-Learn
- `LogisticRegression` supports **multiclass** problems directly, using OvR or multinomial (softmax).
- The default solver is `lbfgs`. Other solvers include `newton-cg`, `liblinear`, `sag` and `saga`.
- `predict_proba()` gives the probability of each class, and each row adds up to 1.
- `argmax` on these probabilities, or simply `predict()`, gives the final class label.
- **Tip:** scikit-learn expects 2D input. For a single sample, use `.reshape(1, -1)`.

### 7. Overfitting and Underfitting
| Problem | Meaning | Also called |
|---------|---------|-------------|
| **Overfitting** | Good on training data, poor on testing data (model too complex) | High variance |
| **Underfitting** | Poor on both training and testing data (model too simple) | High bias |

- **Variance** is how much the predictions change if we train the model again on different data.
- **Bias** is how far the predictions are from the correct values in general.

### 8. Regularization (L2)
- Regularization adds a **penalty for large weights**, which keeps the model simpler and helps prevent overfitting.
- **L2 regularization** (also called weight decay) adds `(λ / 2n) · Σ w²` to the loss.
- A **bigger λ** means stronger regularization and smaller weights. The bias unit is usually not regularized.
- In scikit-learn, the parameter **C = 1/λ**, so a **smaller C means stronger regularization**.
- Feature scaling is important here too, because all features should be on a similar scale.
- The chapter plots the **regularization path**, which shows that the weights shrink towards zero as C decreases.




