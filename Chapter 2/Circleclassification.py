import numpy as np  # numerical arrays and math; used to generate points and compute distances
import pandas as pd  # tabular data; used to inspect the dataset as a DataFrame
import seaborn as sns  # statistical plotting; gives nicer scatter plots with less code
import matplotlib.pyplot as plt  # base plotting library; used to draw the boundary and set labels

n_samples = 100  # number of data points to generate
radius = 1.0  # radius of the circular decision boundary
center = np.array([0.0, 0.0])  # center of the circle (origin)
random_state = 1  # fixed seed so results are reproducible on every run
genrate = np.random.RandomState(random_state)  # seeded random generator (avoids changing global numpy seed)

# Points drawn uniformly from the square that encloses the circle
X = genrate.uniform(low=-1.5, high=1.5, size=(n_samples, 2))  # 100x2 matrix; range > radius so both classes appear

# class 0 -> inside the circle, class 1 -> outside
distance = np.sqrt((X[:, 0] - center[0]) ** 2 + (X[:, 1] - center[1]) ** 2)  # Euclidean distance of each point from center
y = np.where(distance <= radius, 0, 1)  # label: 0 if inside/on circle, 1 if outside

df = pd.DataFrame(X, columns=['feature1', 'feature2'])  # wrap features in a DataFrame for easy viewing
df['distance'] = distance  # add distance column to see why each label was assigned
df['target'] = y  # add class label column
print(df.head(10), "\n")  # show first 10 rows as a quick sanity check
print(df.describe(), "\n")  # summary stats (mean, std, min, max) of each column
print("Class counts:\n", df['target'].value_counts(), "\n")  # check class balance between inside and outside

theta = np.linspace(0, 2 * np.pi, 300)  # 300 angles from 0 to 2π to trace a smooth circle
circle_x = center[0] + radius * np.cos(theta)  # x-coordinates of the boundary circle (parametric form)
circle_y = center[1] + radius * np.sin(theta)  # y-coordinates of the boundary circle (parametric form)

sns.scatterplot(x=X[y == 0, 0], y=X[y == 0, 1],  # plot only class 0 points (boolean mask selects rows)
                color='red', marker='o', label='class 0 (inside)')  # red circles so inside points stand out
sns.scatterplot(x=X[y == 1, 0], y=X[y == 1, 1],  # plot only class 1 points
                color='blue', marker='s', label='class 1 (outside)')  # blue squares, a different shape for clarity
plt.plot(circle_x, circle_y, color='black', linewidth=2, label='boundary')  # draw the true decision boundary
plt.xlabel('feature1')  # label x-axis
plt.ylabel('feature2')  # label y-axis
plt.title('Circle Classification')  # plot title
plt.legend()  # show legend for both classes and the boundary
plt.gca().set_aspect('equal')  # equal axis scaling so the circle doesn't look like an ellipse
plt.show()  # render the figure window
