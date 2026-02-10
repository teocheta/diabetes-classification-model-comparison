import numpy as np


class KNeighborsClassifier:

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def _euclidean_distance(self, x1, x2):
        """Calculează distanța euclidiană între două puncte."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_neighbors(self, x):
        """
        Găsește k cei mai apropiați vecini pentru un punct x.
        Returnează indicii și distanțele.
        """
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((i, dist))

        # Sortăm după distanță și luăm primii k
        distances.sort(key=lambda tup: tup[1])
        neighbors = distances[:self.n_neighbors]
        return neighbors

    def fit(self, X, y):
        """
        Stochează datele de antrenare.
        X: array numpy de forma (n_samples, n_features)
        y: array numpy de forma (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Face predicții pentru fiecare sample din X.
        X: array numpy de forma (n_samples, n_features)
        """
        predictions = []
        X = np.array(X)

        for x in X:
            neighbors = self._get_neighbors(x)

            # Extragem label-urile vecinilor
            neighbor_labels = [self.y_train[idx] for idx, _ in neighbors]

            # Vot majoritar
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            predictions.append(prediction)

        return np.array(predictions)


def train_kneighbors(X_train_scaled, y_train, n_neighbors=5):
    """
    Funcție wrapper pentru compatibilitate cu codul existent.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    return model


def predict_kneighbors(model, X_test_scaled):
    """
    Funcție wrapper pentru compatibilitate cu codul existent.
    """
    return model.predict(X_test_scaled)
