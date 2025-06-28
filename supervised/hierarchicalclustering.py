import numpy as np

class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, X):
        from scipy.cluster.hierarchy import linkage, fcluster
        self.linkage_matrix = linkage(X, method='ward')
        self.labels_ = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')

    def predict(self, X):
        return self.labels_
