import numpy as np
import pandas as pd

X = np.array([[1, 1], [2, 1], [3, 1], [1, 4], [0, 0], [1, 0], [0, 1]], dtype=np.float64)
Y = np.array([1, 0, 0, 0, 1, 1, 1])

df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = Y
print(df, "\n")

# gradient descent needs comparable feature scales
X_std = np.copy(X)
for j in range(X.shape[1]):
    X_std[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()
print("Standardized features:\n", np.round(X_std, 4), "\n")

eta = 0.1
n_iter = 20
random_state = 1
genrate = np.random.RandomState(random_state)
W = genrate.normal(loc=0.0, scale=0.01, size=X.shape[1])
b = np.float64(0)
losses = []
print(f"Initial weights: {W}, bias: {b}\n")

for epoch in range(1, n_iter + 1):
    net_input = np.dot(X_std, W) + b
    output = net_input                      # identity activation
    errors = Y - output
    W = W + eta * 2.0 * np.dot(X_std.T, errors) / X_std.shape[0]
    b = b + eta * 2.0 * errors.mean()
    loss = (errors ** 2).mean()
    losses.append(loss)
    print(f"Epoch {epoch:2d} | MSE={loss:.6f} W={np.round(W, 4)} b={b:+.4f}")

print("\nLosses per epoch:", np.round(losses, 6))
net_input = np.dot(X_std, W) + b
Y_pred = np.where(net_input >= 0.5, 1, 0)
print("Net input values:", np.round(net_input, 4))
print("Predicted values:", Y_pred)
print("Actual values   :", Y)
print("Misclassified samples:", (Y != Y_pred).sum(), "out of", len(Y))
