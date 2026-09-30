import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Load the Wine dataset
wine = load_wine()
print(wine.feature_names)  # Print the feature names
print(wine.target_names)  # Print the target names

X = wine.data[:, [6, 9]]
y = wine.target  

print('Features: ')
print(X)
print('Target: ')
print(y)

# Print the class labels and the number of samples and features
print('Class labels:', np.unique(y))
print('Number of samples:', X.shape[0])
print('Number of features:', X.shape[1])


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Traning of features:')
print(X_train)

print('Traning of Targets:')
print(y_train)

print('Testing of features:')
print(X_test)

print('Testing of Targets:')
print(y_test)

# Print the counts of each class label in the original dataset, training set, and testing set
print("Labels counts in y:", np.bincount(y))
print("Labels counts in y_train:", np.bincount(y_train))
print("Labels counts in y_test:", np.bincount(y_test))


# Standardize the features
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)  

print("Traning of standardize features: ")
print(X_train_std)

print("Testing of standardize features: ")
print(X_test_std)

# Create class of Gradient descent based logistic regression classifier
class LogosticRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        regen = np.random.RandomState(self.random_state)
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ = self.w_ + self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ = self.b_ + self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0])
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


X_train_01 = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01 = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogosticRegressionGD(eta=0.3, n_iter=1000, random_state=1)

lrgd.fit(X_train_01, y_train_01)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Set up marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v') # Define different markers for each class
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # Define different colors for each class
    cmap = ListedColormap(colors[:len(np.unique(y))]) # Create a color map

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Min and max of the first feature with some padding
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Min and max of the second feature with some padding
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution)) # Create a mesh grid covering the feature space
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # Predict the class label for each grid point
    lab = lab.reshape(xx1.shape) # Reshape the predictions to match the mesh grid
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap) # Plot the decision surface
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=wine.target_names[cl], # Use the wine class name in the legend
                    edgecolor='black')

    # Highlight test samples
    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx] # Extract the test samples
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')


# Decision regions can only be drawn in 2D, so train a second Perceptron on two features:
# flavanoids (index 6) and color_intensity (index 9)
plot_decision_regions(X= X_train_01, y=y_train_01, classifier=lrgd)
plt.xlabel('Flavanoids [standardized]')
plt.ylabel('Color intensity [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()