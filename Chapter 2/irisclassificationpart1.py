# os: file/path utilities (imported for convenience; not used below)
import os
# pandas: load and inspect the dataset as a table (DataFrame)
import pandas as pd
# numpy: fast array operations, used to build label/feature arrays
import numpy as np
# seaborn: high-level plotting library for nicer scatter plots
import seaborn as sns
# matplotlib: base plotting library, used for labels, title, and showing the plot
import matplotlib.pyplot as plt

# Download the Iris dataset from the UCI repository; header=None because the file has no column names
df = pd.read_csv('https://archive.ics.uci.edu/ml/'\
                  'machine-learning-databases/iris/iris.data', header=None, encoding='utf-8')

# Save a local copy so the data can be reused offline without downloading again
df.to_csv('iris.csv', index=False)
# Show the first 5 rows to check the data loaded correctly
print(df.head(5))
# Show column types and non-null counts to check for missing values
print(df.info())
# Show summary statistics (mean, std, min, max) of the numeric columns
print(df.describe())

# Take the class label (column 4) of the first 100 rows: only setosa and versicolor (2 classes)
y = df.iloc[0:100, 4].values
# Convert text labels to numbers: setosa -> 0, versicolor -> 1, since the model needs numeric targets
y = np.where(y == 'Iris-setosa', 0, 1)
# Print the numeric target labels to verify the conversion
print("Actual Target Values:\n ", y)

# Take two features for the same 100 rows: sepal length (col 0) and petal length (col 2), so the data can be plotted in 2D
X = df.iloc[0:100, [0, 2]].values
# Print the feature matrix to verify the selected values
print("Feature Values:\n ", X)

# Plot the first 50 samples (setosa) as red circles
sns.scatterplot(x=X[:50, 0], y=X[:50, 1],
                color='red', marker='o', label='setosa')
# Plot the next 50 samples (versicolor) as blue squares to compare the two classes
sns.scatterplot(x = X[50:100, 0], y = X[50:100, 1],
                color='blue', marker='s', label='versicolor')
# Label the x-axis with the first feature name and unit
plt.xlabel('Sepal Length[cm]')
# Label the y-axis with the second feature name and unit
plt.ylabel('Petal Length[cm]')
# Give the plot a title
plt.title('Iris Classification')
# Show the legend so each color/marker is matched to its class
plt.legend()
# Display the plot window
plt.show()
