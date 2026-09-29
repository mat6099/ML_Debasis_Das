import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.optimize._highspy._core as _h
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

print(X.head())
print(y.head())

# Print the class labels and the number of samples and features
print('Class labels:', np.unique(y))
print('Number of samples:', X.shape[0])
print('Number of features:', X.shape[1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Print the counts of each class label in the original dataset, training set, and testing set
print("Labels counts in y:", np.bincount(y))
print("Labels counts in y_train:", np.bincount(y_train))
print("Labels counts in y_test:", np.bincount(y_test))

# Scale features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Train Perceptron
ppn = Perceptron(max_iter=100, eta0=0.01, random_state=1)
ppn.fit(X_train_std, y_train)

# Predict and evaluate
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

acc = accuracy_score(y_test, y_pred)
print("Accuracy of dataset:", acc)

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
                    label=target_names[cl], # Use the class name (malignant / benign) in the legend
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
# worst concave points (index 27) and worst area (index 23)
target_names = load_breast_cancer().target_names # ['malignant', 'benign']
features = [27, 23]
X_train_std_2d = X_train_std[:, features]
X_test_std_2d = X_test_std[:, features]

ppn_2d = Perceptron(max_iter=1000, eta0=0.01, random_state=1)
ppn_2d.fit(X_train_std_2d, y_train)
y_pred_2d = ppn_2d.predict(X_test_std_2d)
acc_2d = accuracy_score(y_test, y_pred_2d)
print('Accuracy (2 features): %.2f' % acc_2d)

# Plot the decision regions for the training and test sets
X_combined_std = np.vstack((X_train_std_2d, X_test_std_2d)) # Combine the standardized training and test sets
y_combined = np.hstack((y_train, y_test)) # Combine the training and test labels
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn_2d,
                      test_idx=range(len(y_train), len(y_combined))) # Test samples come after the training samples
plt.xlabel('Worst concave points [standardized]')
plt.ylabel('Worst area [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
