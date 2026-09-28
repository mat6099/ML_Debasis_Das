import numpy as np  # NumPy for fast vector/matrix math (dot products, random numbers, mean)

class AdalineSGD:  # ADAptive LInear NEuron trained with Stochastic Gradient Descent (update per sample)

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):  # set hyperparameters; no training happens here
        self.eta = eta  # learning rate: step size of each weight update (too big -> diverge, too small -> slow)
        self.n_iter = n_iter  # number of epochs (full passes over the training data)
        self.shuffle = shuffle  # shuffle data every epoch to avoid cycles and make SGD converge better
        self.random_state = random_state  # seed for reproducible weight init and shuffling
        self.w_initialized = False  # flag so partial_fit knows whether weights already exist

    def fit(self, X, y):  # train from scratch on the whole dataset
        self._initialize_weights(X.shape[1])  # create fresh weights, one per feature (X.shape[1] = number of features)
        self.losses_ = []  # store average loss per epoch to check convergence later (e.g. plot it)

        for i in range(self.n_iter):  # repeat training for n_iter epochs
            if self.shuffle:  # only shuffle if the user asked for it
                X, y = self._shuffle(X, y)  # reorder samples randomly so updates are not biased by data order
            losses = []  # collect loss of every sample in this epoch
            for xi, target in zip(X, y):  # loop over samples one at a time (the "stochastic" part of SGD)
                loss = self._update_weights(xi, target)  # update weights using this single sample, get its loss
                losses.append(loss)  # remember this sample's loss
            avg_loss = np.mean(losses)  # average loss over the epoch summarizes how well the model fits
            self.losses_.append(avg_loss)  # save it for later inspection
        return self  # return the object so calls can be chained, e.g. model.fit(X, y).predict(X)

    def partial_fit(self, X, y):  # train further without resetting weights (online / streaming learning)
        if not self.w_initialized:  # weights don't exist yet (first call)
            self._initialize_weights(X.shape[1])  # so create them
        if y.ravel().shape[0] > 1:  # more than one sample was given
            for xi, target in zip(X, y):  # update on each sample separately
                self._update_weights(xi, target)  # one SGD step per sample
        else:  # only a single sample was given
            self._update_weights(X, y)  # do one SGD step directly on it
        return self  # allow method chaining

    def _shuffle(self, X, y):  # helper to shuffle features and labels together
        r = self.rgen.permutation(len(y))  # random order of indices 0..n-1
        return X[r], y[r]  # apply the same order to X and y so each sample keeps its correct label

    def _initialize_weights(self, m):  # create initial weights for m features
        self.rgen = np.random.RandomState(self.random_state)  # seeded random generator for reproducible results
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)  # small random weights break symmetry; w_[0] is unused (bias kept in b_)
        self.w_initialized = True  # mark weights as created
        self.b_ = np.float64(0.0)  # bias term starts at zero; shifts the decision boundary

    def _update_weights(self, xi, target):  # apply the Adaline learning rule for one sample
        net_input = self.net_input(xi)  # z = w·x + b, the weighted sum of inputs
        output = self.activation(net_input)  # linear activation (identity), so output = z
        error = target - output  # how far the prediction is from the true label
        self.w_[1:] += self.eta * xi * error  # gradient step for weights: move in direction that reduces squared error
        self.b_ += self.eta * error  # gradient step for bias
        loss = (error ** 2) / 2.0  # squared error loss for this sample (1/2 simplifies the derivative)
        return loss  # return loss so fit() can track training progress

    def net_input(self, X):  # compute the linear combination of inputs
        return np.dot(X, self.w_[1:]) + self.b_  # dot product of features and weights plus bias

    def activation(self, X):  # activation function of Adaline
        return X  # identity: Adaline learns on the linear output

    def predict(self, X):  # turn continuous output into a class label
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)  # threshold at 0.5 because labels are 0/1
