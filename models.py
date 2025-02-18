
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

class SVMModel():
    def __init__(self, C=100, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel, probability=True)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  


class NaiveBayesModel():
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  


class LogisticRegressionModel():
    def __init__(self):
        self.model = LogisticRegression(
            penalty='l2',
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            class_weight='balanced',
            random_state=42,
            solver='newton-cg',
            max_iter=1000,
            verbose=0,
            warm_start=False,
            n_jobs=None,
            l1_ratio=None
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  


class KNNModel():
    def __init__(self, n_neighbors=3):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]  


class XGBoostModel():
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric='logloss',
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1] 
