import numpy as np  # numerical arrays and math operations
import pandas as pd  # load and handle the CSV dataset as a table
import seaborn as sns  # statistical plotting library (imported but not used here)
from matplotlib import colors  # matplotlib color utilities (not used directly here)
import matplotlib.pyplot as plt  # plotting interface for figures and charts
from matplotlib.colors import ListedColormap  # build a colormap from a fixed list of colors

df = pd.read_csv("iris.csv", header=None, encoding="utf-8")  # read Iris data; file has no header row
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]  # give readable column names

y = df.iloc[0:100, 4].values  # first 100 rows = setosa + versicolor only (binary problem); take species labels
y = np.where(y == 'Iris-setosa', 0, 1)  # convert text labels to numbers: setosa=0, versicolor=1
X = df.iloc[0:100, [0, 2]].values  # use 2 features (sepal length, petal length) so we can plot in 2D


X_std = np.copy(X)  # copy so the original X is not modified
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()  # standardize feature 1 (mean 0, std 1) for stable, faster gradient descent
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()  # standardize feature 2 the same way

class AdalineSGD:  # ADAptive LInear NEuron trained with Stochastic Gradient Descent

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):  # set hyperparameters
        self.eta = eta  # learning rate: step size of each weight update
        self.n_iter = n_iter  # number of epochs (full passes over the training data)
        self.shuffle = shuffle  # shuffle data each epoch to avoid cycles and improve SGD convergence
        self.random_state = random_state  # seed for reproducible weights and shuffling
        self.w_initialized = False  # flag so partial_fit knows whether weights already exist

    def fit(self, X, y):  # train the model from scratch on the full dataset
        self._initialize_weights(X.shape[1])  # create small random weights, one per feature
        self.losses_ = []  # store average loss per epoch to check convergence

        for i in range(self.n_iter):  # repeat training for n_iter epochs
            if self.shuffle:  # optionally randomize sample order
                X, y = self._shuffle(X, y)  # shuffle X and y together so pairs stay matched
            losses = []  # collect loss of each sample in this epoch
            for xi, target in zip(X, y):  # SGD: go through samples one at a time
                loss = self._update_weights(xi, target)  # update weights using this single sample
                losses.append(loss)  # save that sample's loss
            avg_loss = np.mean(losses)  # average loss over the epoch
            self.losses_.append(avg_loss)  # record it for the loss plot
        return self  # return the object so calls can be chained

    def partial_fit(self, X, y):  # online learning: update with new data without resetting weights
        if not self.w_initialized:  # if the model was never trained
            self._initialize_weights(X.shape[1])  # initialize weights first
        if y.ravel().shape[0] > 1:  # if more than one sample was given
            for xi, target in zip(X, y):  # loop over each sample
                self._update_weights(xi, target)  # update weights per sample
        else:  # only a single sample was given
            self._update_weights(X, y)  # update weights once with it
        return self  # return the object for chaining

    def _shuffle(self, X, y):  # helper to randomly reorder the training data
        r = self.rgen.permutation(len(y))  # random permutation of indices 0..n-1
        return X[r], y[r]  # apply the same order to X and y

    def _initialize_weights(self, m):  # m = number of features
        self.rgen = np.random.RandomState(self.random_state)  # random generator with fixed seed for reproducibility
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)  # small random weights break symmetry; w_[0] is unused (bias kept in b_)
        self.w_initialized = True  # mark weights as ready
        self.b_ = np.float64(0.0)  # bias term starts at zero

    def _update_weights(self, xi, target):  # apply the Adaline learning rule for one sample
        net_input = self.net_input(xi)  # weighted sum z = w·x + b
        output = self.activation(net_input)  # linear activation (identity) output
        error = target - output  # difference between true label and prediction
        self.w_[1:] += self.eta * xi * error  # gradient descent step on weights (minimizes squared error)
        self.b_ += self.eta * error  # gradient descent step on bias
        loss = (error ** 2) / 2.0  # squared error loss for this sample (1/2 simplifies the derivative)
        return loss  # return loss so fit() can track it

    def net_input(self, X):  # compute the linear combination of inputs
        return np.dot(X, self.w_[1:]) + self.b_  # z = X·w + b

    def activation(self, X):  # activation function
        return X  # identity: Adaline learns on the linear output

    def predict(self, X):  # predict class labels
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)  # threshold at 0.5 since labels are 0/1



def plot_decision_regions(X, y, classifier, resolution=0.02):  # draw the classifier's decision boundary in 2D
    markers = ('s', 'x', 'o', '^', 'v')  # marker shapes for each class
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # colors for each class
    cmap = ListedColormap(colors[:len(np.unique(y))])  # colormap with one color per class present

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # range of feature 1 with padding
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # range of feature 2 with padding
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),  # build a grid of points covering the plot
                           np.arange(x2_min, x2_max, resolution))  # grid spacing set by resolution
    labels = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # predict class for every grid point
    labels = labels.reshape(xx1.shape)  # reshape predictions back to grid shape for contour plotting
    plt.contourf(xx1, xx2, labels, alpha=0.3, cmap=cmap,  # fill regions by predicted class (semi-transparent)
                 levels=np.arange(-0.5, len(np.unique(y)), 1))  # levels centered on class values so each class gets one color
    plt.xlim(xx1.min(), xx1.max())  # set x-axis limits to the grid
    plt.ylim(xx2.min(), xx2.max())  # set y-axis limits to the grid

    for idx, cl in enumerate(np.unique(y)):  # loop over each class
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],  # plot the actual samples of this class
                    alpha=0.8, c=colors[idx],  # slight transparency and class color
                    marker=markers[idx], label=cl,  # class marker and legend label
                    edgecolor='black')  # black outline makes points easier to see


ada_sgd = AdalineSGD(n_iter=20, eta=0.05)  # create model: 20 epochs, learning rate 0.05
ada_sgd.fit(X_std, y)  # train on standardized features

plt.figure(figsize=(12, 5))  # create a wide figure for two side-by-side plots

plt.subplot(1, 2, 1)  # left plot: decision regions
plot_decision_regions(X_std, y, classifier=ada_sgd)  # draw boundary and data points
plt.title('Adaline - Stochastic Gradient Descent')  # plot title
plt.xlabel('Sepal Length [standardized]')  # x-axis label
plt.ylabel('Petal Length [standardized]')  # y-axis label
plt.legend(loc='upper left')  # show class legend

plt.subplot(1, 2, 2)  # right plot: training loss curve
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')  # average loss per epoch to check convergence
plt.xlabel('Epochs')  # x-axis label
plt.ylabel('Average Loss')  # y-axis label
plt.title('Adaline - Stochastic Gradient Descent')  # plot title

plt.tight_layout()  # adjust spacing so plots don't overlap
plt.show()  # display the figure
