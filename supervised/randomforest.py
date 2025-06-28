import numpy as np
from collections import Counter
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=3, sample_size=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        m = X.shape[0]
        size = self.sample_size if self.sample_size else m
        for _ in range(self.n_estimators):
            indices = np.random.choice(m, size, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTree(depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree_predict(tree, X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

def tree_predict(tree, X):
    # Simple prediction from our DecisionTree class
    predictions = []
    for x in X:
        pred = traverse_tree(x, tree.tree)
        predictions.append(pred)
    return np.array(predictions)

def traverse_tree(x, tree):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    branches = tree[feature]
    val = x[feature]
    if val in branches:
        return traverse_tree(x, branches[val])
    else:
        # Handle unseen values
        return Counter(branches.values()).most_common(1)[0][0]
