import numpy as np

class ElasticNet:
    def __init__(self, lr=0.01, epochs=1000, alpha=1.0, l1_ratio=0.5):
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.W) + self.b
            error = y_pred - y
            gradient = np.dot(X.T, error) + self.alpha * (
                self.l1_ratio * np.sign(self.W) + (1 - self.l1_ratio) * self.W
            )
            self.W -= self.lr * gradient
            self.b -= self.lr * np.sum(error)

    def predict(self, X):
        return np.dot(X, self.W) + self.b

