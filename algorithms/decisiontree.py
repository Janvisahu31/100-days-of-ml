import numpy as np

def entropy(y):
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

class DecisionTree:
    def __init__(self, depth=3):
        self.depth = depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth == self.depth:
            return np.bincount(y).argmax()
        best_feat = np.argmax([self.info_gain(X[:, i], y) for i in range(X.shape[1])])
        tree = {}
        values = np.unique(X[:, best_feat])
        for val in values:
            idx = X[:, best_feat] == val
            tree[val] = self._build_tree(X[idx], y[idx], depth+1)
        return {best_feat: tree}

    def info_gain(self, x, y):
        overall_entropy = entropy(y)
        values, counts = np.unique(x, return_counts=True)
        weighted_entropy = sum([(counts[i]/sum(counts)) * entropy(y[x == val]) for i, val in enumerate(values)])
        return overall_entropy - weighted_entropy
