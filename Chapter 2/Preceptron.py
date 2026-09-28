import numpy as np  # numerical arrays and math (dot product, random numbers)
import pandas as pd  # DataFrame to display the dataset as a neat table

X = np.array([[1, 1], [2, 1], [3, 1], [1, 4], [0, 0]])  # input samples: 5 rows, 2 features each
Y = np.array([1, 0, 0, 0, 1])  # true class label (0 or 1) for each sample

df = pd.DataFrame(X, columns=['feature1', 'feature2'])  # wrap features in a table for readable printing
df['target'] = Y  # add labels as a column so features and target are shown together
print(df, "\n")  # show the training data before learning starts

eta = 0.01  # learning rate: how big each weight correction step is
n_iter = 20  # number of epochs: full passes over the training data
random_state = 1  # seed so the random initial weights are reproducible
genrate = np.random.RandomState(random_state)  # seeded random generator (same numbers every run)
W = genrate.normal(loc=0.0, scale=0.01, size=X.shape[1])  # small random weights, one per feature, to break symmetry
b = np.float64(0)  # bias starts at 0; shifts the decision boundary away from the origin
errors = []  # stores misclassification count per epoch to track convergence
print(f"Initial weights: {W}, bias: {b}\n")  # show starting parameters

for epoch in range(1, n_iter + 1):  # repeat training for n_iter epochs
    error = 0  # reset misclassification counter for this epoch
    for x, target in zip(X, Y):  # visit each sample with its true label
        net = b  # start net input with the bias term
        for w_i, x_i in zip(W, x):  # loop over each weight/feature pair
            net = net + w_i * x_i  # accumulate weighted sum: net = w·x + b
        y_hat = 1 if net >= 0.0 else 0  # unit step activation: predict class 1 if net >= 0
        update = eta * (target - y_hat)  # perceptron rule: zero if correct, ±eta if wrong
        for i, (w_i, x_i) in enumerate(zip(W, x)):  # loop to update each weight
            W[i] = w_i + update * x_i  # move weight toward correct side, scaled by feature value
        b = b + update  # bias updated like a weight whose input is always 1
        error = error + int(update != 0.0)  # count a mistake whenever an update happened
        print(f"  epoch {epoch:2d} | x={x} target={target} pred={y_hat} "  # log per-sample progress
              f"update={update:+.3f} W={np.round(W, 4)} b={b:+.3f}")  # show current weights and bias
    errors.append(error)  # save this epoch's mistakes to see if learning converges
    print(f"Epoch {epoch:2d} finished -> misclassifications: {error}\n")  # epoch summary

print("Errors per epoch:", errors)  # 0 errors at the end means data was separated
net_input = np.dot(X, W) + b  # compute net input for all samples at once with learned weights
Y_pred = np.where(net_input >= 0, 1, 0)  # apply step function to get final predictions
print("Net input Values:", net_input)  # raw scores before thresholding
print("Predicted values:", Y_pred)  # model's predicted class labels
print("Actual values   :", Y)  # true labels, to compare against predictions
