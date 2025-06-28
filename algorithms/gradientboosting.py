import numpy as np

class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        from decision_tree import DecisionTree
        m = X.shape[0]
        self.f0 = np.mean(y)
        residuals = y - self.f0
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(depth=3)
            tree.fit(X, residuals)
            pred = np.array(tree_predict(tree, X))
            residuals -= self.learning_rate * pred
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            pred += s
