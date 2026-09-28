import numpy as np                                  # NumPy: fast array math (dot products, mean, std, meshgrid)
import pandas as pd                                 # pandas: read the CSV file into a table (DataFrame)
import seaborn as sns                               # seaborn: statistical plotting library (imported but not used below)
from matplotlib import colors                       # matplotlib colors module (not used; the name is overwritten inside plot_decision_regions)
import matplotlib.pyplot as plt                     # pyplot: draw figures, scatter plots, line plots
from matplotlib.colors import ListedColormap        # ListedColormap: build a colormap from a fixed list of colors, one per class

# Load the Iris dataset; header=None because the file has no header row
df = pd.read_csv("iris.csv", header=None, encoding="utf-8")
# Give the 5 columns readable names so the data is easier to understand
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Take the class labels (column 4) of the first 100 rows: rows 0-49 are setosa, 50-99 are versicolor
# Using only 2 classes because Adaline is a binary classifier
y = df.iloc[0:100, 4].values
# Convert text labels into numbers: setosa -> 0, versicolor -> 1 (the model needs numeric targets)
y = np.where(y == 'Iris-setosa', 0, 1)
# Take only 2 features (column 0 = sepal length, column 2 = petal length) so the decision boundary can be drawn in 2D
X = df.iloc[0:100, [0, 2]].values


# Make a copy so the original X is not modified
X_std = np.copy(X)
# Standardize feature 1: (value - mean) / std -> mean 0, std 1
# Why: gradient descent converges much faster and more stably when features are on the same scale
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
# Standardize feature 2 in the same way
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

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



def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')                          # marker shape per class (square, x, circle, ...)
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')       # color per class
    cmap = ListedColormap(colors[:len(np.unique(y))])            # colormap with exactly as many colors as there are classes

    # Plot range for each feature, extended by 1 on each side so points are not on the border
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Build a dense grid of points covering the whole plot area (spacing = resolution)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # Flatten the grid into a list of (x1, x2) points and predict the class of every grid point
    labels = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Reshape the predictions back to the grid shape so they can be drawn as a 2D surface
    labels = labels.reshape(xx1.shape)
    # Fill the regions with class colors -> shows the decision boundary; alpha=0.3 keeps it transparent
    # levels are set at -0.5, 0.5, ... so each integer class falls in its own color band
    plt.contourf(xx1, xx2, labels, alpha=0.3, cmap=cmap,
                 levels=np.arange(-0.5, len(np.unique(y)), 1))
    # Fix the axis limits to the grid range
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Draw the actual training samples on top, one class at a time with its own color and marker
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],   # select only the samples belonging to class cl
                    alpha=0.8, c=colors[idx],           # slightly transparent, class color
                    marker=markers[idx], label=cl,      # class marker; label is used by the legend
                    edgecolor='black')                  # black outline makes points easy to see


# Create the model: 20 epochs, learning rate 0.05 (works well because data is standardized)
ada_gd = AdalineGD(n_iter=20, eta=0.05)
# Train on the standardized features
ada_gd.fit(X_std, y)

# Create one figure 12 inches wide x 5 inches tall to hold two side-by-side plots
plt.figure(figsize=(12, 5))

# Left plot (1 row, 2 columns, position 1): decision regions
plt.subplot(1, 2, 1)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('Sepal Length [standardized]')
plt.ylabel('Petal Length [standardized]')
plt.legend(loc='upper left')                            # show which color/marker is class 0 and class 1

# Right plot (position 2): loss vs. epochs, to check that training converges (loss should decrease)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean-squared-error')
plt.title('Adaline - Gradient Descent')

plt.tight_layout()                                      # adjust spacing so titles/labels don't overlap
plt.show()                                              # display the figure window

