import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

n_samples = 100
radius = 1.0
center = np.array([0.0, 0.0])
random_state = 1
genrate = np.random.RandomState(random_state)

# Points drawn uniformly from the square that encloses the circle
X = genrate.uniform(low=-1.5, high=1.5, size=(n_samples, 2))

# class 0 -> inside the circle, class 1 -> outside
distance = np.sqrt((X[:, 0] - center[0]) ** 2 + (X[:, 1] - center[1]) ** 2)
y = np.where(distance <= radius, 0, 1)

df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['distance'] = distance
df['target'] = y
print(df.head(10), "\n")
print(df.describe(), "\n")
print("Class counts:\n", df['target'].value_counts(), "\n")

theta = np.linspace(0, 2 * np.pi, 300)
circle_x = center[0] + radius * np.cos(theta)
circle_y = center[1] + radius * np.sin(theta)

sns.scatterplot(x=X[y == 0, 0], y=X[y == 0, 1],
                color='red', marker='o', label='class 0 (inside)')
sns.scatterplot(x=X[y == 1, 0], y=X[y == 1, 1],
                color='blue', marker='s', label='class 1 (outside)')
plt.plot(circle_x, circle_y, color='black', linewidth=2, label='boundary')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.title('Circle Classification')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()
