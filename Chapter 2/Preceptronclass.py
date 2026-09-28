import numpy as np  # NumPy for fast vector/matrix math (dot products, random numbers)

class Preceptron:  # Perceptron classifier: learns a linear decision boundary for binary labels
    def __init__(self, eta=0.001, n_iter=70, random_state=1):  # Constructor: store hyperparameters
        self.eta = eta  # Learning rate: how big each weight update step is
        self.n_iter = n_iter  # Number of epochs (full passes over the training data)
        self.random_state = random_state  # Seed so random weight init is reproducible

    def fit(self, X, y):  # Train the model on features X (samples x features) and labels y
        regen = np.random.RandomState(self.random_state)  # Seeded random generator for reproducible results
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # Small random weights to break symmetry (note: 1+ gives one extra weight; bias is stored separately in b_, so size should be X.shape[1])
        self.b_ = np.float64(0.0)  # Bias term starts at zero; shifts the decision boundary
        self.errors_ = []  # Stores misclassification count per epoch to check convergence

        for _ in range(self.n_iter):  # Repeat training for n_iter epochs
            errors = 0  # Reset error counter for this epoch
            for xi, target in zip(X, y):  # Loop over each sample and its true label
                update = self.eta * (target - self.predict(xi))  # Perceptron rule: step is non-zero only if prediction is wrong
                self.w_ += update * xi  # Move weights toward correct classification for this sample
                self.b_ += update  # Adjust bias the same way (its input is always 1)
                errors += int(update != 0.0)  # Count a mistake whenever an update happened
            self.errors_.append(errors)  # Save epoch's errors to plot learning progress later
        return self  # Return the fitted object so calls can be chained

    def net_input(self, X):  # Compute the weighted sum z = w·x + b
        return np.dot(X, self.w_) + self.b_  # Dot product of inputs and weights plus bias

    def predict(self, X):  # Predict class label(s) for input X
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # Unit step function: 1 if z >= 0, else 0
