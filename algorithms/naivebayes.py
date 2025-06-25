import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(X)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = -0.5 * np.sum(np.log(2 * np.pi * self.var[c]))
            class_conditional -= 0.5 * np.sum(((x - self.mean[c])**2) / self.var[c])
            posteriors.append(prior + class_conditional)
        return self.classes[np.argmax(posteriors)]
