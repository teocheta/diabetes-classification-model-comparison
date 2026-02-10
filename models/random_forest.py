import numpy as np
import random

#arbore care învață reguli de tip dacă–atunci
class DecisionTree:
    def _init_(self, max_depth=10, min_samples_split=2, random_state=None):
        self.max_depth = max_depth #Adâncimea maximă a arborelui
        self.min_samples_split = min_samples_split #Nr min de exemple necesare ca un nod să mai fie împărțit
        self.random_state = random_state #seed
        self.tree = None

    def _gini_impurity(self, y): #y = lista etichetelor (0 și 1)
        """Calculează Gini impurity pentru un set de label-uri.
        cât de „amestecate” sunt clasele într-un nod
         Gini = 0 → toate datele sunt din aceeași clasă (nod „curat”)
         Gini mare → clasele sunt amestecate (nod „neclar”)
        Arborele vrea Gini cât mai mic.
        """
        if len(y) == 0:
            return 0

        counts = np.bincount(y) #Numără câte valori sunt din fiecare clasă.
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini #probabilitatea ca un element ales aleator să fie clasificat greșit, prsctic cât de „rău” e nodul

    def _best_split(self, X, y, feature_indices): #X – datele, y – etichetele
        """
        Găsește cea mai bună bună regulă de tip dacă feature ≤ prag pentru un nod.
        feature_indices: lista de indici de features disponibili pentru split
        """
        #pornim cu cea mai proastă impuritate posibilă
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_idx in feature_indices:
            # Obținem valorile unice pentru acest feature
            feature_values = np.unique(X[:, feature_idx])

            for threshold in feature_values:
                # Împărțim datele
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # Dacă toate datele merg într-o singură parte, split inutil → îl sar
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculăm Gini impurity pentru ambele părți
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])

                # Calculăm Gini weighted (Impuritatea este ponderată în funcție de câte date sunt în fiecare nod)
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)

                weighted_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini

                # daca splitul e mai bun, il salvam
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _build_tree(self, X, y, depth=0, feature_indices=None):
        """
        Construiește recursiv arborele de decizie.
        """
        #La rădăcină, arborele poate folosi toate feature-urile daca nu se primesc explicit
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))

        # Condiții de oprire
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            # Returnăm clasa majoritară
            return {'class': int(np.bincount(y).argmax())}

        # Dacă nodul este deja pur, îl transform direct într-o frunză
        if len(np.unique(y)) == 1:
            return {'class': int(y[0])}

        # Alegem regula care minimizează Gini impurity
        best_feature, best_threshold, best_gini = self._best_split(X, y, feature_indices)

        # Dacă nu există o împărțire bună, iau decizia majoritară
        if best_feature is None:
            return {'class': int(np.bincount(y).argmax())}

        # Datele sunt împărțite în funcție de regula aleasă
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Construiesc sub-arborii recursiv, până ajung la frunze
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_indices)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_indices)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X, y):
        """Antrenează arborele de decizie."""
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, tree):
        """Face predicție pentru un singur sample."""
        # Dacă ajung într-o frunză, întorc clasa
        if 'class' in tree:
            return tree['class']
        # Parcurg arborele recursiv până ajung la o frunză
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

#creează multi arbori și îi combină
class RandomForestClassifier:
    """
    Implementare Random Forest de la zero folosind bagging și feature sampling.
    """

    def _init_(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 max_features=None, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators # Numărul de arbori
        self.max_depth = max_depth # Adâncimea maximă a fiecărui arbore
        self.min_samples_split = min_samples_split # Nr min de date necesare pentru a face o împărțire
        self.max_features = max_features # Nr de coloane folosite de fiecare arbore (Fiecare arbore vede doar o parte din feature-uri, ca să nu învețe toți la fel)
        self.random_state = random_state # seed
        self.n_jobs = n_jobs # pentru paralelizare
        self.trees = [] # aici se stocheaza fiecare arbore de decizie
        self.feature_indices_per_tree = [] # ce feature-uri a folosit fiecare arbore

    def _bootstrap_sample(self, X, y):
        """
        face ca fiecare arbore să fie antrenat pe un set de date diferit
        """
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        construiesc toți arborii din Random Forest
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []
        self.feature_indices_per_tree = []

        for i in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X, y)

            # Selectare aleatoare de features
            feature_indices = random.sample(range(n_features), self.max_features)
            X_boot_selected = X_boot[:, feature_indices]

            # Antrenare arbore
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i
            )
            tree.fit(X_boot_selected, y_boot)

            self.trees.append(tree)
            self.feature_indices_per_tree.append(feature_indices)

    def predict(self, X):
        """
        colectează predicțiile tuturor arborilor și alege clasa votată de majoritate
        """
        X = np.array(X)
        all_predictions = []

        # Pentru fiecare arbore, folosesc aceleași feature-uri ca la antrenare
        for tree, feature_indices in zip(self.trees, self.feature_indices_per_tree):
            X_selected = X[:, feature_indices]
            predictions = tree.predict(X_selected)
            all_predictions.append(predictions)

        # Vot majoritar
        all_predictions = np.array(all_predictions)
        final_predictions = []
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            final_predictions.append(int(np.bincount(votes).argmax()))

        return np.array(final_predictions)


def train_random_forest(X_train_scaled, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    return rf_model


def predict_random_forest(rf_model, X_test_scaled):
    return rf_model.predict(X_test_scaled)