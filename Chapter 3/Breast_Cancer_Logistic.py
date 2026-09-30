import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
target_names = load_breast_cancer().target_names

print(X.head())
print(y.head())

X = X.iloc[:, [27, 23]]  # DataFrame needs .iloc for positional column selection
print(X.head())

# Convert to NumPy arrays so boolean-mask indexing like X[y == cl, 0] works later
X, y = X.to_numpy(), y.to_numpy()


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

# Train a Perceptron model
lr = LogisticRegression(C=100.0, solver='lbfgs')
lr.fit(X_train_std, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

# Find Accuracy i.e evalution of model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy)


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
                    label=target_names[cl], # Use the breast cancer class name in the legend
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


# Plot the decision regions for the training and test sets
X_combined_std = np.vstack((X_train_std, X_test_std)) # Combine the standardized training and test sets
y_combined = np.hstack((y_train, y_test)) # Combine the training and test labels
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=lr,
                      test_idx=range(len(y_train), len(y_combined))) # Test samples come after the training samples
plt.xlabel('Flavanoids [standardized]')
plt.ylabel('Color intensity [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()