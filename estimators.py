# Copyright (c) Gradient Institute. All rights reserved.
# Licensed under the Apache 2.0 License.
"""Convenience estimators for causal estimation."""

from typing import NamedTuple, Union

import numpy as np
import pandas as pd

from scipy import linalg, stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.utils.validation import check_is_fitted


class RegressionStatisticalResults(NamedTuple):
    """Statistical results object for linear regressors.

    Attributes
    ----------
    beta: float or ndarray
        the regression coefficients
    std_err: float or ndarray
        The standard error of the coefficients
    t_stat: float or ndarray
        The t-statistics for the regression coefficients
    p_value: float or ndarray
        The p-value of the two-sided t-test on the coefficients. The null
        hypothesis is that beta = 0, and the alternate hypothesis is beta != 0.
    dof: float
        The degrees of freedom used to compute the t-test.
    """

    beta: Union[float, np.ndarray]
    std_err: Union[float, np.ndarray]
    t_stat: Union[float, np.ndarray]
    p_value: Union[float, np.ndarray]
    dof: float

    def __repr__(self) -> str:
        """Return string representation of StatisticalResults."""
        reprs = f"""Statistical results:
            beta =
                {self.beta},
            s.e.(beta) =
                {self.std_err}
            t-statistic(s):
                {self.t_stat}
            p-value(s):
                {self.p_value}
            Degrees of freedom: {self.dof}
            """
        return reprs


class _StatMixin:
    def model_statistics(self):
        """Get the coefficient statistics for this estimator."""
        check_is_fitted(self, attributes=["coef_", "coef_se_"])
        stats = RegressionStatisticalResults(
            beta=self.coef_,
            std_err=self.coef_se_,
            dof=self.dof_,
            t_stat=self.t_,
            p_value=self.p_,
        )
        return stats


class LinearRegressionStat(LinearRegression, _StatMixin):
    """Scikit learn's LinearRegression estimator with coefficient stats."""

    def fit(self, X, y, sample_weight=None):
        """Fit linear regression model to data.

        TODO: complete docstring
        """
        super().fit(X, y, sample_weight)
        n, d = X.shape
        self.dof_ = n - d
        shp = (d,) if np.isscalar(self._residues) else (d, len(self._residues))
        s2 = ((self._residues / self.dof_) * np.ones(shp)).T
        self.coef_se_ = np.sqrt(linalg.pinv(X.T @ X).diagonal() * s2)
        self.t_ = self.coef_ / self.coef_se_
        self.p_ = (1.0 - stats.t.cdf(np.abs(self.t_), df=self.dof_)) * 2
        return self


class BayesianRidgeStat(BayesianRidge, _StatMixin):
    """Scikit learn's BayesianRidge estimator with coefficient stats."""

    def fit(self, X, y, sample_weight=None):
        """Fit bayesian ridge estimator to data.

        TODO: complete docstring
        """
        super().fit(X, y, sample_weight)
        n, d = X.shape
        self.dof_ = n - d  # NOTE: THIS IS AN UNDERESTIMATE
        self.coef_se_ = np.sqrt(self.sigma_.diagonal())
        # NOTE: THIS IS NOT USING THE PROPER POSTERIOR
        self.t_ = self.coef_ / self.coef_se_
        self.p_ = (1.0 - stats.t.cdf(np.abs(self.t_), df=self.dof_)) * 2
        return self
