from typing import Tuple

import pandas as pd
from sklearn.utils import resample


def upsample(X: pd.DataFrame, y: pd.DataFrame, minority_val=1, majority_val=0, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Upsample minority class to correct class imbalance."""

    df = pd.concat([X, y], axis=1)
    df_min = df[df.iloc[:, -1] == minority_val]
    df_maj = df[df.iloc[:, -1] == majority_val]
    df_min_upsampled = resample(
        df_min, replace=True, n_samples=df_maj.shape[0], random_state=42)
    df_upsampled = pd.concat([df_min_upsampled, df_maj])

    if verbose:
        print('Balanced class counts: ')
        print(df_upsampled.iloc[:, -1].value_counts())

    return df_upsampled.iloc[:, :-1], df_upsampled.iloc[:, -1]
