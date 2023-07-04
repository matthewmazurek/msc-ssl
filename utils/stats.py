from statistics import NormalDist, mean, variance

import numpy as np
from sklearn.utils import resample

# def bootstrap_ci(stat, data, replicates=100, alpha=0.05):
#     """Returns the bootstrap (1-alpha)100% confidence interval for a given statistic"""
#     B = [stat(*resample(*data)) for i in range(replicates)]

#     m = mean(B)
#     se = (variance(B)/replicates)**(.5)
#     z = NormalDist().inv_cdf(1-alpha/2)

#     return dict(
#         stat=stat(*data),
#         normal=[m - se * z, m + se * z],
#         percentile=[np.quantile(B, alpha/2), np.quantile(B, 1-alpha/2)],
#         params=dict(replicates=replicates, alpha=alpha)
#     )


def bootstrap_ci(stat, data, replicates=100, alpha=0.05):
    """Returns the bootstrap (1-alpha)100% confidence interval for a given statistic"""
    data, bootstrap_data = get_bootstrap_data(data, replicates)
    return bootstrap_estimate(stat, data, bootstrap_data, alpha)


def get_bootstrap_data(data, replicates):
    return (data, [resample(*data) for i in range(replicates)])


def bootstrap_estimate(stat, data, bootstrap_data, alpha=0.05, test=0):

    B = [stat(*data) for data in bootstrap_data]

    n = len(B)
    m = mean(B)
    se = (variance(B)/n)**(.5)
    z = NormalDist().inv_cdf(1-alpha/2)

    return dict(
        stat=stat(*data),
        se=se,
        normal=[m - se * z, m + se * z],
        p=1-NormalDist().cdf(abs((m-test))/se),
        percentile=[np.quantile(B, alpha/2), np.quantile(B, 1-alpha/2)],
        params=dict(replicates=n, alpha=alpha)
    )


class estimate_ci:
    """A helper function to print consistent estimate and confidence intervals"""

    def __init__(self, estimate, ci, sig_dig=3):
        self.estimate = estimate
        self.ci = ci
        self.sig_dig = sig_dig

    def __str__(self) -> str:
        return f'{self.estimate:.{self.sig_dig}f} ({self.ci[0]:.{self.sig_dig}f}–{self.ci[1]:.{self.sig_dig}f})'
