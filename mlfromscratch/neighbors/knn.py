import numpy as np
from collections import Counter


class KNN:
    def __init__(self, n_neighbors=5, distance_metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.empty(X.shape[0])

        for i, x in enumerate(X):
            print(f'{i=}, {x.shape=}')
            distances = self._get_distances(x)
            y_pred[i] = self._get_majority_vote(distances)

        return y_pred

    def _get_distances(self, x: np.ndarray) -> np.ndarray:
        if self.distance_metric == 'euclidean':
            print(f'{self.X.shape=}, {x.shape=}')
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.X - x), axis=1)
        else:
            raise ValueError('Invalid distance metric specified.')

        return distances

    def _get_majority_vote(self, distances: np.ndarray) -> int:
        print(f'{distances.shape=}')
        k_nearest_indices = np.argsort(distances)[: self.n_neighbors]
        print(f'{k_nearest_indices=}')
        k_nearest_labels = self.y[k_nearest_indices]
        print(f'{k_nearest_labels=}')
        majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]

        return majority_vote
