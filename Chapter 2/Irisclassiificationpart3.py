import os  # OS utilities (file paths); not strictly needed here
import pandas as pd  # Load and handle tabular data (CSV)
import numpy as np  # Fast numerical arrays and math
import seaborn as sns  # Statistical plotting (imported, not used below)
from matplotlib import colors  # Color utilities (shadowed later by local 'colors' tuple)
import matplotlib.pyplot as plt  # Plotting API for figures and charts
from matplotlib.colors import ListedColormap  # Build a colormap from a fixed list of colors

df = pd.read_csv('iris.csv', header=None, encoding='utf-8')  # Read Iris data; no header row in file

y = df.iloc[0:100, 4].values  # Take class labels of first 100 rows (Setosa + Versicolor only)
y = np.where(y == 'Iris-setosa', 0, 1)  # Convert text labels to binary: Setosa=0, Versicolor=1
print("Actual Target Values:\n ", y)  # Show the encoded target labels

X = df.iloc[0:100, [0, 2]].values  # Pick 2 features: sepal length (col 0) and petal length (col 2) for 2D plotting
print("Feature Values:\n ", X)  # Show the feature matrix


class Preceptron:  # Simple Perceptron binary classifier (Rosenblatt)
    def __init__(self, eta=0.001, n_iter=70, random_state=1):  # Set hyperparameters
        self.eta = eta  # Learning rate: step size of each weight update
        self.n_iter = n_iter  # Number of passes (epochs) over the training data
        self.random_state = random_state  # Seed so weight initialization is reproducible

    def fit(self, X, y):  # Train the model on features X and labels y
        regen = np.random.RandomState(self.random_state)  # Seeded random generator for repeatable results
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=X.shape[1])  # Small random weights, one per feature (avoids symmetry)
        self.b_ = np.float64(0.0)  # Bias term starts at zero
        self.errors_ = []  # Store misclassification count per epoch to monitor convergence

        for _ in range(self.n_iter):  # Loop over epochs
            errors = 0  # Reset error count for this epoch
            for xi, target in zip(X, y):  # Go through each sample and its true label
                update = self.eta * (target - self.predict(xi))  # Perceptron rule: nonzero only if prediction is wrong
                self.w_ += update * xi  # Adjust weights toward correct classification
                self.b_ += update  # Adjust bias (shifts decision boundary)
                errors += int(update != 0.0)  # Count a mistake whenever an update happened
            self.errors_.append(errors)  # Save this epoch's number of mistakes
        return self  # Return trained model (allows method chaining)

    def net_input(self, X):  # Compute weighted sum z = w·x + b
        return np.dot(X, self.w_) + self.b_  # Linear combination of inputs plus bias

    def predict(self, X):  # Predict class label(s)
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # Unit step function: z>=0 -> 1, else 0


pp = Preceptron(eta=0.001, n_iter=70, random_state=1)  # Create Perceptron with chosen hyperparameters
pp.fit(X, y)  # Train it on the Iris subset

def plot_decision_regions(X, y, classifier, resolution=0.02):  # Visualize how the classifier splits the 2D feature space
    markers = ('s', 'x', 'o', '^', 'v')  # Marker shapes for each class
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # Colors for each class
    cmap = ListedColormap(colors[:len(np.unique(y))])  # Colormap with as many colors as there are classes

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Range of feature 1 with padding
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Range of feature 2 with padding
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),  # Build a fine grid of points covering the plot
                           np.arange(x2_min, x2_max, resolution))  # 'resolution' sets grid spacing
    labels = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # Predict class for every grid point
    labels = labels.reshape(xx1.shape)  # Reshape predictions back to grid shape for contour plotting
    plt.contourf(xx1, xx2, labels, alpha=0.3, cmap=cmap,  # Fill regions by predicted class (semi-transparent)
                 levels=np.arange(-0.5, len(np.unique(y)), 1))  # Level boundaries centered on class labels 0,1,...
    plt.xlim(xx1.min(), xx1.max())  # Set x-axis limits to grid range
    plt.ylim(xx2.min(), xx2.max())  # Set y-axis limits to grid range

    for idx, cl in enumerate(np.unique(y)):  # Loop over each class to draw its samples
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],  # Plot samples of this class
                    alpha=0.8, c=colors[idx],  # Slight transparency and class color
                    marker=markers[idx], label=cl,  # Class-specific marker and legend label
                    edgecolor='black')  # Black outline so points stand out


plot_decision_regions(X, y, classifier=pp)  # Draw decision regions of the trained Perceptron
plt.xlabel('Sepal Length [cm]')  # Label x-axis (feature 1)
plt.ylabel('Petal Length [cm]')  # Label y-axis (feature 2)
plt.legend(loc='upper left')  # Show class legend in top-left corner
plt.title('Perceptron Decision Regions')  # Add plot title
plt.show()  # Display the figure
