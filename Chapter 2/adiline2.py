import numpy as np               # NumPy: fast array math (dot products, random numbers, np.where)
import pandas as pd              # pandas: read the CSV file into a table (DataFrame)
import seaborn as sns            # seaborn: statistical plotting library (imported but not used below)
import matplotlib.pyplot as plt  # matplotlib: draw the loss-vs-epoch plots

# Load the Iris dataset. header=None because the file has no header row,
# so pandas should not treat the first data row as column names.
df = pd.read_csv("iris.csv", header=None, encoding="utf-8")
# Give the 5 columns readable names so the data is easier to understand.
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Take the class label (column 4) of the first 100 rows.
# The first 100 rows contain only two species (Setosa and Versicolor),
# which gives us a binary (two-class) problem suitable for Adaline.
y = df.iloc[0:100, 4].values
# Convert text labels to numbers: Setosa -> 0, Versicolor -> 1.
# Adaline needs numeric targets to compute the error (y - output).
y = np.where(y == 'Iris-setosa', 0, 1)
print("Actual Target Values:\n ", y)  # show the numeric labels to verify the conversion

# Select two features: sepal length (column 0) and petal length (column 2).
# Using only two features keeps the problem simple and easy to visualise in 2D.
X = df.iloc[0:100, [0, 2]].values
print("Feature Values:\n ", X)  # show the feature matrix (100 rows x 2 columns)


class AdalineGD:
    """Adaptive Linear Neuron classifier (batch gradient descent)."""

    def __init__(self, eta=0.01, n_iter=70, random_state=1):
        self.eta = eta                    # learning rate: how big each weight update step is
        self.n_iter = n_iter              # number of epochs (full passes over the training data)
        self.random_state = random_state  # seed so the random initial weights are reproducible

    def fit(self, X, y):
        # Random number generator with a fixed seed -> same results every run.
        regen = np.random.RandomState(self.random_state)
        # Initialise one weight per feature with small random values (mean 0, std 0.01).
        # Small non-zero values break symmetry and keep the initial output near 0.
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # Bias (intercept) starts at 0; float64 so it can be updated with decimal values.
        self.b_ = np.float64(0.0)
        # Store the loss of every epoch so we can check whether training converges.
        self.losses_ = []

        for _ in range(self.n_iter):  # repeat training for n_iter epochs
            # Weighted sum of inputs for ALL samples at once: z = X·w + b.
            net_input = self.net_input(X)
            # Apply activation (identity for Adaline), so output = z.
            output = self.activation(net_input)
            # Error between true label and continuous output for every sample.
            errors = y - output
            # Gradient descent update of weights. The MSE loss is L = mean((y - z)^2),
            # its gradient w.r.t. w is -2 * X^T·errors / n, so we move opposite to it.
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            # Gradient descent update of bias: gradient w.r.t. b is -2 * mean(errors).
            self.b_ += self.eta * 2.0 * errors.mean()
            # Mean squared error for this epoch — the quantity Adaline minimises.
            loss = (errors ** 2).mean()
            # Save it to plot the learning curve later.
            self.losses_.append(loss)
        return self  # return the trained object so calls can be chained, e.g. AdalineGD().fit(X, y)

    def net_input(self, X):
        # Linear combination of features and weights plus bias: z = X·w + b.
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X  # identity: Adaline learns on the linear output

    def predict(self, X):
        # Turn the continuous output into a class label with a threshold of 0.5
        # (midway between the two targets 0 and 1): >= 0.5 -> class 1, else class 0.
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# Create one figure with two side-by-side plots to compare two learning rates.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Model 1: larger learning rate (0.01). On unscaled features this is too large,
# so the loss grows (diverges) instead of shrinking.
ada1 = AdalineGD(n_iter=15, eta=0.01).fit(X, y)
# Plot log10 of the loss, because the diverging values become huge and a
# log scale keeps the curve readable.
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')                   # x-axis: training epoch number
ax[0].set_ylabel('log(Mean-squared-error)')  # y-axis: log of the loss
ax[0].set_title('Adaline - Learning rate 0.01')

# Model 2: very small learning rate (0.0001). The loss decreases, but slowly,
# showing that a too-small rate needs many epochs to converge.
ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
# Plot the raw loss (no log needed because the values stay small).
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')                # x-axis: training epoch number
ax[1].set_ylabel('Mean-squared-error)')   # y-axis: loss value
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.tight_layout()  # adjust spacing so titles and labels don't overlap
plt.show()          # display the figure window
