import numpy as np
import random


class XGBoostTree:
    """
    Implementare simplificată de Decision Tree pentru XGBoost.
    Folosește gradient boosting cu loss-ul logistic.
    """

    def __init__(self, max_depth=3, min_samples_split=2, learning_rate=0.1, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.tree = None

    def _sigmoid(self, x):
        """Funcția sigmoid."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def _gradient(self, y, y_pred):
        """Gradient pentru logistic loss."""
        return y_pred - y

    def _hessian(self, y, y_pred):
        """Hessian pentru logistic loss."""
        return y_pred * (1 - y_pred)

    def _calculate_leaf_value(self, gradients, hessians, lambda_reg=1.0):
        """
        Calculează valoarea optimă pentru un nod frunză folosind formula XGBoost.
        """
        if len(gradients) == 0:
            return 0.0

        sum_gradients = np.sum(gradients)
        sum_hessians = np.sum(hessians)

        if sum_hessians == 0:
            return 0.0

        # Formula XGBoost: -sum(gradients) / (sum(hessians) + lambda)
        leaf_value = -sum_gradients / (sum_hessians + lambda_reg)
        return leaf_value * self.learning_rate

    def _gain(self, gradients, hessians, lambda_reg=1.0, gamma=0.0):
        """
        Calculează gain-ul pentru un split.
        """
        sum_gradients = np.sum(gradients)
        sum_hessians = np.sum(hessians)

        if sum_hessians == 0:
            return 0.0

        gain = (sum_gradients ** 2) / (sum_hessians + lambda_reg) - gamma
        return gain

    def _best_split(self, X, gradients, hessians, feature_indices):
        """
        Găsește cea mai bună împărțire bazată pe gain.
        """
        best_gain = float('-inf')
        best_feature = None
        best_threshold = None

        total_gain = self._gain(gradients, hessians)

        for feature_idx in feature_indices:
            feature_values = np.unique(X[:, feature_idx])

            for threshold in feature_values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gradients = gradients[left_mask]
                left_hessians = hessians[left_mask]
                right_gradients = gradients[right_mask]
                right_hessians = hessians[right_mask]

                left_gain = self._gain(left_gradients, left_hessians)
                right_gain = self._gain(right_gradients, right_hessians)

                gain = left_gain + right_gain - total_gain

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, gradients, hessians, depth=0, feature_indices=None):
        """
        Construiește recursiv arborele.
        """
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))

        # Condiții de oprire
        if depth >= self.max_depth or len(gradients) < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            return {'leaf': True, 'value': leaf_value}

        # Găsim cea mai bună împărțire
        best_feature, best_threshold, best_gain = self._best_split(X, gradients, hessians, feature_indices)

        if best_feature is None or best_gain <= 0:
            leaf_value = self._calculate_leaf_value(gradients, hessians)
            return {'leaf': True, 'value': leaf_value}

        # Împărțim datele
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Construim sub-arborii
        left_tree = self._build_tree(
            X[left_mask], gradients[left_mask], hessians[left_mask],
            depth + 1, feature_indices
        )
        right_tree = self._build_tree(
            X[right_mask], gradients[right_mask], hessians[right_mask],
            depth + 1, feature_indices
        )

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X, y_pred_prev, y_true):
        """
        Antrenează arborele folosind gradient-urile și hessian-urile.
        """
        X = np.array(X)
        y_pred_prev = np.array(y_pred_prev)
        y_true = np.array(y_true)

        # Calculăm gradient-urile și hessian-urile
        gradients = self._gradient(y_true, y_pred_prev)
        hessians = self._hessian(y_true, y_pred_prev)

        self.tree = self._build_tree(X, gradients, hessians)

    def _predict_sample(self, x, tree):
        """Face predicție pentru un singur sample."""
        if tree['leaf']:
            return tree['value']

        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        """Face predicții pentru un set de samples."""
        predictions = []
        X = np.array(X)
        for x in X:
            predictions.append(self._predict_sample(x, self.tree))
        return np.array(predictions)


class XGBClassifier:
    """
    Implementare simplificată XGBoost pentru clasificare binară.
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3,
                 subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.trees = []
        self.feature_indices_per_tree = []
        self.base_score = None

    def _sigmoid(self, x):
        """Funcția sigmoid."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Antrenează modelul folosind gradient boosting.
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        n_features_to_use = max(1, int(self.colsample_bytree * n_features))

        # Inițializare cu probabilitatea medie
        self.base_score = np.mean(y)
        y_pred = np.full(n_samples, self.base_score)

        self.trees = []
        self.feature_indices_per_tree = []

        for i in range(self.n_estimators):
            # Subsampling
            if self.subsample < 1.0:
                n_samples_to_use = max(1, int(self.subsample * n_samples))
                sample_indices = np.random.choice(n_samples, size=n_samples_to_use, replace=False)
                X_sub = X[sample_indices]
                y_sub = y[sample_indices]
                y_pred_sub = y_pred[sample_indices]
            else:
                X_sub = X
                y_sub = y
                y_pred_sub = y_pred

            # Feature sampling
            feature_indices = random.sample(range(n_features), n_features_to_use)
            X_sub_selected = X_sub[:, feature_indices]

            # Antrenare arbore
            tree = XGBoostTree(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state + i
            )

            # Transformăm y_pred în probabilități pentru calculul gradient-urilor
            y_pred_proba = self._sigmoid(y_pred_sub)
            tree.fit(X_sub_selected, y_pred_proba, y_sub)

            self.trees.append(tree)
            self.feature_indices_per_tree.append(feature_indices)

            # Update predicții pentru toate samples-urile
            X_selected = X[:, feature_indices]
            tree_predictions = tree.predict(X_selected)
            y_pred = y_pred + tree_predictions

    def predict_proba(self, X):
        """
        Returnează probabilitățile pentru clasa pozitivă.
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # Start cu base score
        y_pred = np.full(n_samples, self.base_score)

        # Adăugăm predicțiile din fiecare arbore
        for tree, feature_indices in zip(self.trees, self.feature_indices_per_tree):
            X_selected = X[:, feature_indices]
            tree_predictions = tree.predict(X_selected)
            y_pred = y_pred + tree_predictions

        # Transformăm în probabilități
        probabilities = self._sigmoid(y_pred)
        return probabilities

    def predict(self, X):
        """
        Returnează predicțiile binare (0 sau 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


def train_xgboost(X_train_scaled, y_train):
    """
    Funcție wrapper pentru compatibilitate cu codul existent.
    """
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_scaled, y_train)
    return xgb_model


def predict_xgboost(xgb_model, X_test_scaled):
    """
    Funcție wrapper pentru compatibilitate cu codul existent.
    """
    return xgb_model.predict(X_test_scaled)
