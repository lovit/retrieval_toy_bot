import numpy as np
from sklearn.metrics import pairwise_distances

class FullSearchIndexer:

    def __init__(self, x):
        self.x = x

    def kneighbors(self, query, n_neighbors=10, max_distance=0.5):
        dist = pairwise_distances(query, self.x, metric='cosine')[0]
        idx = dist.argsort()

        # filtering with n_neighbors
        if n_neighbors > 0:
            idx = idx[:n_neighbors]
        dist_ = dist[idx]

        # filtering with max distance
        distant = np.where(dist_ > max_distance)[0]
        if distant.shape[0] > 0:
            idx = idx[:distant[0]]
            dist_ = dist_[:distant[0]]

        return dist_, idx

    def save(self, fname):
        raise NotImplemented

    def load(self, fname):
        raise NotImplemented