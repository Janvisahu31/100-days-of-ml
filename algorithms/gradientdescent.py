import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    W = np.zeros(X.shape[1])
    b = 0
    m = len(y)
    for _ in range(epochs):
        y_pred = np.dot(X, W) + b
        error = y_pred - y
        W -= lr * (1/m) * np.dot(X.T, error)
        b -= lr * (1/m) * np.sum(error)
    return W, b
