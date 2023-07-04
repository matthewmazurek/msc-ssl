# feature helper functions
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import pandas as pd


class Ftype(Enum):
    dichotomous = auto()
    categorical = auto()
    continuous = auto()


@dataclass
class Feature():
    name: str
    type: Ftype
    target: bool = False


class FeatureList():

    def __init__(self, fts: list = None):
        self.features = []
        if fts:
            self.features = fts

    def append(self, ft: Feature):
        self.features.append(ft)

    def sort(self):
        self.features = sorted(self.features, key=lambda i: (i.type))

    def lookup(self, ft_type: Ftype = None, target: bool = False):
        if ft_type:
            return [ft.name for ft in self.features if ft.type == ft_type and ft.target == target]

        # categorical variables last to make one-hot coding labeling easier
        ft_type_list = [Ftype.continuous, Ftype.dichotomous, Ftype.categorical]
        return [ft.name for ft in self.features for type in ft_type_list if ft.target == target and ft.type == type]

    def to_table(self):
        return pd.DataFrame(
            {'feature': ft.name, 'type': ft.type.name} for ft in self.features
        ).set_index('feature')
