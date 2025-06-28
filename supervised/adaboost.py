import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        m = len(y)
        w = np.ones(m) / m
        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            from decision_tree import DecisionTree
            stump = DecisionTree(depth=1)
            stump.fit(X, y)
            pred = np.array(tree_predict(stump, X))
            error = np.sum(w * (pred != y))
            if error == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            w = w * np.exp(-alpha * y * pred)
            w /= np.sum(w)
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        pred = sum(alpha * np.array(tree_predict(stump, X)) for stump, alpha in zip(self.models, self.alphas))
        return np.sign(pred)

def tree_predict(tree, X):
    predictions = []
    for x in X:
        pred = traverse_tree(x, tree.tree)
        predictions.append(pred)
    return predictions

def traverse_tree(x, tree):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    branches = tree[feature]
    val = x[feature]
    if val in branches:
        return traverse_tree(x, branches[val])
    else:
        return list(branches.values())[0]
