import os  # OS utilities (file paths, env); imported for convenience, not used below
import pandas as pd  # pandas: load and inspect the tabular Iris dataset
import numpy as np  # numpy: fast array math for weights, dot products, labels
import seaborn as sns  # seaborn: nicer high-level plotting on top of matplotlib
import matplotlib.pyplot as plt  # matplotlib: set axis labels/title and show the plot

df = pd.read_csv('iris.csv', header=None, encoding='utf-8')  # Read CSV into a DataFrame; header=None because the file has no column names

print(df.head(5))  # Show first 5 rows to quickly check the data loaded correctly
print(df.info())  # Show column types and non-null counts to spot missing/wrong-type data
print(df.describe())  # Show summary stats (mean, std, min, max) of numeric columns

y = df.iloc[0:100, 4].values  # Take class labels (column 4) of first 100 rows = setosa + versicolor only (binary problem)
y = np.where(y == 'Iris-setosa', 0, 1)  # Convert text labels to numbers: setosa -> 0, versicolor -> 1 (perceptron needs numeric targets)
print("Actual Target Values:\n ", y)  # Print encoded labels to verify the conversion

X = df.iloc[0:100, [0, 2]].values  # Pick 2 features (sepal length, petal length) of first 100 rows as NumPy array
print("Feature Values:\n ", X)  # Print feature matrix to verify the input data


class Preceptron:  # Perceptron classifier: a single neuron with a step activation
    def __init__(self, eta=0.001, n_iter=70, random_state=1):  # Constructor storing hyperparameters
        self.eta = eta  # Learning rate: how big each weight update step is
        self.n_iter = n_iter  # Number of epochs (full passes over the training data)
        self.random_state = random_state  # Seed so random weight init is reproducible

    def fit(self, X, y):  # Train the perceptron on features X and labels y
        regen = np.random.RandomState(self.random_state)  # Seeded random generator for repeatable results
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=X.shape[1])  # Init weights with small random values (one per feature) to break symmetry
        self.b_ = np.float64(0.0)  # Init bias to 0; float64 so it can be updated with fractional values
        self.errors_ = []  # List to record misclassifications per epoch (to track learning progress)

        for _ in range(self.n_iter):  # Repeat training for n_iter epochs
            errors = 0  # Reset misclassification count for this epoch
            for xi, target in zip(X, y):  # Loop over each sample and its true label
                update = self.eta * (target - self.predict(xi))  # Perceptron rule: eta * (true - predicted); 0 if correct
                self.w_ += update * xi  # Move weights toward correct classification for this sample
                self.b_ += update  # Adjust bias the same way (its input is always 1)
                errors += int(update != 0.0)  # Count a mistake whenever an update happened
            self.errors_.append(errors)  # Save epoch's error count for plotting convergence
        return self  # Return the trained object (allows chaining like pp.fit(X, y).predict(...))

    def net_input(self, X):  # Compute weighted sum z = w·x + b
        return np.dot(X, self.w_) + self.b_  # Dot product of inputs and weights plus bias

    def predict(self, X):  # Predict class label for input(s)
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # Step function: z >= 0 -> class 1, else class 0


pp = Preceptron(eta=0.001, n_iter=70, random_state=1)  # Create perceptron with chosen learning rate, epochs and seed
pp.fit(X, y)  # Train the model on the Iris features and labels
sns.lineplot(x=range(1, len(pp.errors_) + 1),  # x-axis: epoch numbers starting at 1
             y=pp.errors_, marker='o',  # y-axis: misclassifications per epoch; 'o' marks each epoch point
             color='blue', label='Misclassifications')  # Line color and legend label
plt.xlabel('Epochs')  # Label x-axis
plt.ylabel('Number of misclassifications')  # Label y-axis
plt.title('Perceptron Learning Algorithm')  # Plot title
plt.show()  # Display the plot window (errors should drop to 0 if data is linearly separable)
