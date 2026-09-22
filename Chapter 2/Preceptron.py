import numpy as np
import pandas as pd

X = np.array([[1, 1], [2, 1], [3, 1], [1, 4], [0, 0]])
Y = np.array([1, 0, 0, 0, 1])

df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = Y
print(df, "\n")

eta = 0.01
n_iter = 20
random_state = 1
genrate = np.random.RandomState(random_state)
W = genrate.normal(loc=0.0, scale=0.01, size=X.shape[1])
b = np.float64(0)
errors = []
print(f"Initial weights: {W}, bias: {b}\n")

for epoch in range(1, n_iter + 1):
    error = 0
    for x, target in zip(X, Y):
        net = b
        for w_i, x_i in zip(W, x):
            net = net + w_i * x_i
        y_hat = 1 if net >= 0.0 else 0
        update = eta * (target - y_hat)
        for i, (w_i, x_i) in enumerate(zip(W, x)):
            W[i] = w_i + update * x_i
        b = b + update
        error = error + int(update != 0.0)
        print(f"  epoch {epoch:2d} | x={x} target={target} pred={y_hat} "
              f"update={update:+.3f} W={np.round(W, 4)} b={b:+.3f}")
    errors.append(error)
    print(f"Epoch {epoch:2d} finished -> misclassifications: {error}\n")

print("Errors per epoch:", errors)
net_input = np.dot(X, W) + b
Y_pred = np.where(net_input >= 0, 1, 0)
print("Net input Values:", net_input)
print("Predicted values:", Y_pred)
print("Actual values   :", Y)
