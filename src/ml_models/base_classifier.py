import numpy as np
from sklearn.linear_model import LogisticRegression



class BaseLogisticClassifier():

    def __init__(self, **kwargs):

        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            **kwargs
        )


    def fit(self,X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

