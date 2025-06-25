
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.W) + self.b
            error = y_pred - y
            self.W -= self.lr * (1/self.m) * np.dot(X.T, error)
            self.b -= self.lr * (1/self.m) * np.sum(error)

    def predict(self, X):
        return np.dot(X, self.W) + self.b