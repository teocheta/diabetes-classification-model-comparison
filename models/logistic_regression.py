import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Evităm overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _loss(self, y_true, y_pred):
        # Evităm log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Antrenare model folosind gradient descent.
        X: array numpy de forma (n_samples, n_features)
        y: array numpy de forma (n_samples,)
        """
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Inițializare weights și bias
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)

            # Calculare loss
            loss = self._loss(y, y_pred)

            # Calculare gradient
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)

            # Update weights și bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # early stopping dacă loss-ul nu se mai schimbă
            if i % 100 == 0:
                pass

    def predict_proba(self, X):
        """Returnează probabilitățile pentru clasa pozitivă."""
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X):
        """Returnează predicțiile binare (0 sau 1)."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


def train_logistic_regression(X_train_scaled, y_train, random_state=42):
    """
    Funcție wrapper
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model


def predict_logistic_regression(model, X_test_scaled):
    """
    Funcție wrapper
    """
    return model.predict(X_test_scaled)
