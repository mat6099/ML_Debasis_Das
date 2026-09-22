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


class Preceptron:
    def __init__(self, eta=0.001, n_iter=70, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        regen = np.random.RandomState(self.random_state)
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


pp = Preceptron(eta=0.001, n_iter=70, random_state=1)
pp.fit(X, y)
sns.lineplot(x=range(1, len(pp.errors_) + 1), 
             y=pp.errors_, marker='o', 
             color='blue', label='Misclassifications')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron Learning Algorithm')
plt.show()