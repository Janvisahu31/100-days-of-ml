import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y):
        X_poly = self._transform(X)
        from linear_regression import LinearRegression
        self.model = LinearRegression()
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self._transform(X)
        return self.model.predict(X_poly)

    def _transform(self, X):
        return np.hstack([X ** i for i in range(1, self.degree + 1)])
