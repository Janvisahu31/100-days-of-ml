import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        for _ in range(self.epochs):
            linear = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(linear)
            error = y_pred - y
            self.W -= self.lr * (1/self.m) * np.dot(X.T, error)
            self.b -= self.lr * (1/self.m) * np.sum(error)

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.W) + self.b) >= 0.5
