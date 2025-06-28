import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X_centered = X - np.mean(X, axis=0)
        cov = np.cov(X_centered.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eig_vals)[::-1]
        self.components = eig_vecs[:, sorted_idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered, self.components)
