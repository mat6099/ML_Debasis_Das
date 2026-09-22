import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/'\
                  'machine-learning-databases/iris/iris.data', header=None, encoding='utf-8')

print(df.head(5))
print(df.info())
print(df.describe())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
print("Actual Target Values:\n ", y)

X = df.iloc[0:100, [0, 2]].values
print("Feature Values:\n ", X)

sns.scatterplot(x=X[:50, 0], y=X[:50, 1],
                color='red', marker='o', label='setosa')
sns.scatterplot(x = X[50:100, 0], y = X[50:100, 1],
                color='blue', marker='s', label='versicolor')
plt.xlabel('Sepal Length[cm]')
plt.ylabel('Petal Length[cm]')
plt.title('Iris Classification')
plt.legend()
plt.show()