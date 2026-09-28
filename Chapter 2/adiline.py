import numpy as np                     # numerical arrays and vector math (dot products, mean, std)
import pandas as pd                    # DataFrame to display the dataset in a readable table

# training data: 7 samples, 2 features each; float64 so standardization keeps decimals
X = np.array([[1, 1], [2, 1], [3, 1], [1, 4], [0, 0], [1, 0], [0, 1]], dtype=np.float64)
Y = np.array([1, 0, 0, 0, 1, 1, 1])    # class labels (0/1) — the targets Adaline learns to fit

df = pd.DataFrame(X, columns=['feature1', 'feature2'])  # wrap features in a table for display
df['target'] = Y                       # add labels as a column so features and target show together
print(df, "\n")                        # show the dataset before training

# gradient descent needs comparable feature scales
X_std = np.copy(X)                     # copy so the original X is not overwritten
for j in range(X.shape[1]):            # loop over each feature column
    X_std[:, j] = (X[:, j] - X[:, j].mean()) / X[:, j].std()  # z-score: mean 0, std 1 → faster, stable convergence
print("Standardized features:\n", np.round(X_std, 4), "\n")  # show scaled features

eta = 0.1                              # learning rate: step size of each weight update
n_iter = 20                            # number of epochs (full passes over the data)
random_state = 1                       # seed so results are reproducible
genrate = np.random.RandomState(random_state)  # seeded random generator
W = genrate.normal(loc=0.0, scale=0.01, size=X.shape[1])  # small random weights, one per feature (breaks symmetry)
b = np.float64(0)                      # bias starts at zero
losses = []                            # stores MSE per epoch to track convergence
print(f"Initial weights: {W}, bias: {b}\n")  # show starting parameters

for epoch in range(1, n_iter + 1):     # full-batch gradient descent loop
    net_input = np.dot(X_std, W) + b   # weighted sum z = X·W + b for all samples at once
    output = net_input                      # identity activation
    errors = Y - output                # residuals: how far continuous output is from true labels
    W = W + eta * 2.0 * np.dot(X_std.T, errors) / X_std.shape[0]  # step against MSE gradient w.r.t. W
    b = b + eta * 2.0 * errors.mean()  # step against MSE gradient w.r.t. b
    loss = (errors ** 2).mean()        # mean squared error — the cost Adaline minimizes
    losses.append(loss)                # record loss to check that it decreases
    print(f"Epoch {epoch:2d} | MSE={loss:.6f} W={np.round(W, 4)} b={b:+.4f}")  # progress per epoch

print("\nLosses per epoch:", np.round(losses, 6))  # full loss history (should be decreasing)
net_input = np.dot(X_std, W) + b       # final net input using trained weights
Y_pred = np.where(net_input >= 0.5, 1, 0)  # threshold at 0.5 (midpoint of labels 0/1) to get class
print("Net input values:", np.round(net_input, 4))  # continuous outputs before thresholding
print("Predicted values:", Y_pred)     # predicted class labels
print("Actual values   :", Y)          # true labels for comparison
print("Misclassified samples:", (Y != Y_pred).sum(), "out of", len(Y))  # count of wrong predictions
