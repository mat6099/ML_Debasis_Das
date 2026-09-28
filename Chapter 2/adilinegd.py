import numpy as np
import pandas as pd

class AdalineGD:
    """Adaptive Linear Neuron classifier (batch gradient descent)."""

    def __init__(self, eta=0.01, n_iter=70, random_state=1):
        self.eta = eta                      # learning rate: size of each weight-update step (too big -> diverge, too small -> slow)
        self.n_iter = n_iter                # number of epochs: how many full passes over the training data
        self.random_state = random_state    # seed for the random generator so results are reproducible

    def fit(self, X, y):
        # Random number generator with a fixed seed -> same initial weights every run
        regen = np.random.RandomState(self.random_state)
        # Initialize weights with small random values (not all zeros) so the learning has a non-trivial starting point
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # Initialize bias (intercept) to 0
        self.b_ = np.float64(0.0)
        # List to record the loss after each epoch, used later to check convergence
        self.losses_ = []

        for _ in range(self.n_iter):                    # repeat training for n_iter epochs
            net_input = self.net_input(X)               # z = X·w + b for ALL samples at once (batch gradient descent)
            output = self.activation(net_input)         # linear activation: output = z (continuous, not 0/1)
            errors = y - output                         # difference between true label and model output
            # Weight update from the gradient of MSE: w += eta * 2/n * X^T · errors
            # Moves weights in the direction that reduces the mean squared error
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            # Bias update from the gradient of MSE: b += eta * 2 * mean(errors)
            self.b_ += self.eta * 2.0 * errors.mean()
            # Mean squared error for this epoch = the cost function Adaline minimizes
            loss = (errors ** 2).mean()
            self.losses_.append(loss)                   # store loss so we can plot it vs. epochs
        return self                                     # return the trained object (allows chaining, e.g. model.fit(...).predict(...))

    def net_input(self, X):
        # Weighted sum of inputs plus bias: z = w1*x1 + w2*x2 + b
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X  # identity: Adaline learns on the linear output

    def predict(self, X):
        # Threshold step: if linear output >= 0.5 predict class 1, else class 0
        # 0.5 is used because targets are 0 and 1, so 0.5 is the midpoint between them
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

