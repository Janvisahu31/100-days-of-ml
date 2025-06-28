import numpy as np

class LDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        n_features = X.shape[1]
        mean_overall = np.mean(X, axis=0)
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            Sw += np.dot((X_c - mean_c).T, (X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            Sb += n_c * np.dot(mean_diff, mean_diff.T)
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
        sorted_indices = np.argsort(-eig_vals.real)
        self.linear_discriminants = eig_vecs[:, sorted_indices].real

    def transform(self, X, n_components):
        return np.dot(X, self.linear_discriminants[:, :n_components])
