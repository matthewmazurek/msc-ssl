from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.features import FeatureList
from utils.preprocessing import upsample


class DataFactory:

    def __init__(self, name, fts: FeatureList = None):
        self.name = name
        self.fts = fts if fts else FeatureList()

    def load(self, path):
        self.df = pd.read_csv(path, na_values="?")
        return self

    def register_features(self, fts: FeatureList):
        self.fts = fts

    def get_targets(self):
        return self.df[self.fts.lookup(target=True)]

    def get_predictors(self):
        return self.df[self.fts.lookup(target=False)]

    def get_partitions(self, balanced=False, **kwargs):
        # partition the data
        X, y = self.get_predictors(), self.get_targets()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **kwargs
        )

        # upsample training sets to counter class imbalance
        if balanced:
            X_train, y_train = upsample(X_train, y_train)

        return X_train, X_test, y_train, y_test
