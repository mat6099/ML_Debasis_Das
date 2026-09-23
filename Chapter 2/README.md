# Chapter 2 — Training Simple Machine Learning Algorithms for Classification


## Introduction

This chapter is the real starting point for building machine learning models by hand. It teaches you the very first classification algorithms in ML history and shows how they actually work under the hood. Instead of just calling a library function, you write the algorithms yourself using plain Python and NumPy. This helps you understand *why* they work, not just *how* to use them.

The chapter has two big ideas: the **Perceptron** and **Adaline** (Adaptive Linear Neuron). Both are simple "single neuron" models used for binary classification (telling apart two classes). You learn to train them on the famous **Iris flower dataset** and draw pictures of their decision boundaries.

## What You Will Learn

- The early history of machine learning and how artificial neurons came from studying the human brain.
- How the **Perceptron learning rule** works and how to code it as a clean Python class.
- How to train a model on real data (the Iris dataset) and check if it is learning.
- How to draw **decision regions** to see how the model separates classes.
- What **Adaline** is and how it improves on the Perceptron using **gradient descent**.
- Why **feature scaling (standardization)** makes training faster and more stable.
- The difference between **batch gradient descent**, **stochastic gradient descent (SGD)**, and **mini-batch learning**, and what **online learning** means.

## Section-wise Overview

### 1. Artificial Neurons — A Short History
The chapter starts with the **McCulloch-Pitts neuron**, introduced by Warren McCulloch and Walter Pitts in their 1943 paper *"A Logical Calculus of the Ideas Immanent in Nervous Activity"* (Bulletin of Mathematical Biophysics 5:115–133). This is one of the earliest ideas of an artificial neuron. Scientists tried to copy how a brain cell (neuron) fires. A neuron takes many inputs, adds them up with some weights, and gives an output. If the total signal crosses a limit (threshold), the neuron "fires" (output +1), otherwise it stays off (output -1). This simple on/off idea is the base of everything that follows.

### 2. Rosenblatt's Perceptron
Frank Rosenblatt introduced the perceptron in his 1958 *Psychological Review* paper *"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"* (65(6):386–408), building on his earlier 1957 Cornell Aeronautical Laboratory Report 85-460-1. He built on the McCulloch-Pitts neuron and gave a way for the neuron to **learn** its own weights from data. This is the **perceptron learning rule**. The idea is simple:
- Start with small or zero weights.
- For each training example, guess the class.
- If the guess is wrong, nudge the weights a little in the right direction.
- Repeat many times (called epochs) until the mistakes reduce.

The perceptron only works well when the two classes can be separated by a straight line (this is called **linearly separable** data). If they cannot be separated by a line, it will never fully settle down.

### 3. Implementing a Perceptron in Python (Object-Oriented API)
Here you build a `Perceptron` class using NumPy. It follows a clean, reusable design similar to scikit-learn. The class has methods like:
- `fit()` — trains the model (updates weights over several epochs).
- `net_input()` — calculates the weighted sum of inputs.
- `predict()` — gives the final class label (+1 or -1).

You also track the number of wrong guesses in each epoch to see if learning is working.

### 4. Training on the Iris Dataset
The chapter uses the **Iris dataset**, introduced by Ronald A. Fisher in his 1936 paper *"The Use of Multiple Measurements in Taxonomic Problems"* (Annals of Eugenics 7(2):179–188). It is a classic beginner dataset of flower measurements. To keep it simple:
- Only the first 100 samples are used.
- Only two flower types are picked (Setosa and Versicolor) to make it a two-class problem.
- Only two features are used (so we can plot in 2D).

You load the data with **pandas**, prepare it, and train the perceptron on it.

### 5. Plotting Decision Regions
A helper function draws the areas of the graph where the model predicts each class. This gives a nice visual picture of how the straight-line boundary splits the two flower types. This step uses **Matplotlib**.

### 6. Adaptive Linear Neuron (Adaline)
Next comes **Adaline**, a smarter cousin of the perceptron. The big difference is *how it learns*. The perceptron updates weights based on the final yes/no label. Adaline instead uses the raw continuous output (before the threshold) to measure error. This makes learning smoother and is closer to how modern neural networks are trained.

### 7. Minimizing the Cost Function with Gradient Descent
Adaline defines a **cost function** (also called a loss function) that measures total error. Training means making this cost as small as possible. The method used is **gradient descent** — think of it as slowly walking downhill to reach the lowest point of an error valley. At each step, weights move a little in the direction that reduces the cost. The step size is set by the **learning rate**. If it is too big, you overshoot; too small, and learning is very slow.

### 8. Feature Scaling (Standardization)
Real-world features have different ranges (for example, one column may be 0–10 and another 0–1000). This confuses gradient descent. **Standardization** fixes this by rescaling each feature to have mean 0 and standard deviation 1. After standardization, Adaline learns much faster and more reliably.

### 9. Stochastic Gradient Descent (SGD) and Mini-Batch Learning
Normal (batch) gradient descent uses the *whole* dataset for each weight update, which is slow for big data. **Stochastic gradient descent** updates the weights using *one example at a time*, which is much faster and often good enough. **Mini-batch** learning is a middle path — it uses a small group of examples per update. The chapter also explains **online learning**, where the model keeps learning from new data as it arrives, without retraining from scratch.

## Key Formulas Explained Simply

- **Net input (weighted sum):**
  `z = w1*x1 + w2*x2 + ... + wm*xm + b`
  This is just adding up all inputs multiplied by their importance (weights), plus a bias term `b`.

- **Perceptron prediction (threshold):**
  `output = +1 if z >= 0, else -1`
  A simple on/off decision.

- **Perceptron weight update rule:**
  `Δw_j = η * (y_true - y_pred) * x_j`
  If the guess is correct, the change is zero. If wrong, weights shift by an amount set by the learning rate `η` (eta).

- **Adaline cost function (Sum of Squared Errors):**
  `J(w) = (1/2) * Σ (y_true - φ(z))²`
  This adds up all the squared errors. Adaline tries to make this number as small as possible. (Note: the 2022 PyTorch/Scikit-Learn book frames this as Mean Squared Error, which is the same idea averaged over samples.)

- **Standardization formula:**
  `x' = (x - μ) / σ`
  Subtract the mean `μ`, then divide by the standard deviation `σ`.

## Code / Notebook Items Covered

- `Perceptron` — the perceptron classifier class.
- `plot_decision_regions()` — helper function to visualise decision boundaries.
- `AdalineGD` — Adaline trained with full-batch gradient descent.
- `AdalineSGD` — Adaline trained with stochastic gradient descent (includes shuffling and partial updates).
- Data loading and preparation of the Iris dataset with pandas.
- Plots for training errors/cost per epoch and decision regions with Matplotlib.

## Key Takeaways

- The perceptron and Adaline are the "hello world" of machine learning — simple, but they teach the core idea of **learning weights from data**.
- The perceptron uses a hard yes/no rule; Adaline uses a smooth cost function, which is the foundation of modern deep learning.
- **Gradient descent** is the main engine that trains most machine learning models today.
- **Feature scaling** is not optional — it can make a huge difference in training speed and stability.
- Choosing between **batch, mini-batch, and stochastic** gradient descent is a trade-off between speed and stability.
- Writing algorithms from scratch builds strong intuition before moving to ready-made libraries.

## Prerequisites and Libraries Used

**Prerequisites:**
- Basic Python (functions, classes, loops).
- Basic maths: what a mean, sum, and simple equation is.
- A little idea of vectors/arrays helps but is not compulsory.

**Libraries used:**
- **Python 3**
- **NumPy** — for fast maths on arrays.
- **pandas** — for loading and handling the dataset.
- **Matplotlib** — for plotting graphs and decision regions.
