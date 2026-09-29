import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Extract the features (petal length and petal width) and target labels (species)
X = iris.data[:, [2, 3]]
y = iris.target

# Print the class labels and the number of samples and features
print('Class labels:', np.unique(y))
print('Number of samples:', X.shape[0])
print('Number of features:', X.shape[1])

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Print the counts of each class label in the original dataset, training set, and testing set
print("Labels counts in y:", np.bincount(y))
print("Labels counts in y_train:", np.bincount(y_train))
print("Labels counts in y_test:", np.bincount(y_test))

# Standardize the features by removing the mean and scaling to unit variance
sc = StandardScaler()

# fit_transform the training data but transform the testing data
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Create a Perceptron classifier with a learning rate of 0.1 and a random state for reproducibility
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# Make predictions on the test set
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Set up marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v') # Define different markers for each class
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # Define different colors for each class
    cmap = ListedColormap(colors[:len(np.unique(y))]) # Create a color map

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Define the min and max values for the first feature (petal length) with some padding
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Define the min and max values for the second feature (petal width) with some padding
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), # Create a mesh grid for the first feature (petal length) using the defined min and max values and the specified resolution
                           np.arange(x2_min, x2_max, resolution)) # Create a mesh grid for the second feature (petal width) using the defined min and max values and the specified resolution
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # Predict the class labels for each point in the mesh grid
    lab = lab.reshape(xx1.shape) # Reshape the predicted labels to match the shape of the mesh grid
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap) # Plot the decision surface using contourf with the predicted labels and the defined color map
    plt.xlim(xx1.min(), xx1.max()) # Set the x-axis limits to the min and max values of the first feature (petal length)
    plt.ylim(xx2.min(), xx2.max()) # Set the y-axis limits to the min and max values of the second feature (petal width)


    # Plot all samples
    for idx, cl in enumerate(np.unique(y)): # Loop through each unique class label in the target labels
        plt.scatter(x=X[y == cl, 0],     # X coordinate of the samples belonging to the 0 class label
                    y=X[y == cl, 1],     # Y coordinate of the samples belonging to the 1 class label
                    alpha=0.8,           # Set the transparency of the points to 0.8
                    c=colors[idx],       # colour of the points based on the class label
                    marker=markers[idx], # marker style of the points based on the class label
                    label=cl,            # label for the class in the legend
                    edgecolor='black')   # Set the edge color of the points to black

    # Highlight test samples
    if test_idx: # If test indices are provided
        
        X_test, y_test = X[test_idx, :], y[test_idx] # Extract the test samples and their corresponding labels
        
        plt.scatter(X_test[:, 0],
                    X_test[:, 1], # Plot the test samples
                    c='none', # Set the color of the test samples to none
                    edgecolor='black', # Set the edge color of the test samples to black
                    alpha=1.0, # Set the transparency of the test samples to 1.0
                    linewidth=1, # Set the line width of the edges of the test samples to 1
                    marker='o', # Use a circle marker for the test samples
                    s=100, # Set the size of the test sample markers to 100
                    label='test set') # Label for the test samples in the legend


# Plot the decision regions for the training and test sets
X_combined_std = np.vstack((X_train_std, X_test_std)) # Combine the standardized training and test sets
y_combined = np.hstack((y_train, y_test)) # Combine the training and test labels
# Plot the decision regions using the combined data and the trained classifier
plot_decision_regions(X=X_combined_std,
                      y=y_combined, 
                      classifier=ppn, 
                      test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]') # Set the x-axis label
plt.ylabel('Petal width [standardized]') # Set the y-axis label
plt.tight_layout() # Adjust the layout of the plot to make it tight
plt.show() # Display the plot