import numpy as np

class IsolationTree:
    def __init__(self, max_depth, current_depth=0):
        self.max_depth = max_depth
        self.current_depth = current_depth

    def fit(self, X):
        if self.current_depth >= self.max_depth or len(X) <= 1:
            self.is_leaf = True
            self.size = len(X)
            return
        q = np.random.randint(X.shape[1])
        min_val, max_val = np.min(X[:, q]), np.max(X[:, q])
        if min_val == max_val:
            self.is_leaf = True
            self.size = len(X)
            return
        p = np.random.uniform(min_val, max_val)
        left_mask = X[:, q] < p
        self.q = q
        self.p = p
        self.left = IsolationTree(self.max_depth, self.current_depth + 1)
        self.left.fit(X[left_mask])
        self.right = IsolationTree(self.max_depth, self.current_depth + 1)
        self.right.fit(X[~left_mask])
        self.is_leaf = False

    def path_length(self, x):
        if self.is_leaf:
            return self.current_depth
        if x[self.q] < self.p:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)

class IsolationForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth

    def fit(self, X):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample = X[np.random.choice(X.shape[0], size=X.shape[0]//2, replace=False)]
            tree = IsolationTree(self.max_depth)
            tree.fit(X_sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        avg_path_lengths = np.array([np.mean([tree.path_length(x) for tree in self.trees]) for x in X])
        c = 2 * (np.log(len(X) - 1) + 0.5772) - (2 * (len(X) - 1) / len(X))
        return np.exp(-avg_path_lengths / c)
