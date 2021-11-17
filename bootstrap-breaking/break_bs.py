"""Examples which break bootstrap estimation.
"""
from typing import Union
from functools import partial
from collections.abc import Callable
import math

import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
import matplotlib.pyplot as plt


def subsample_estimator(
    subsampler: Callable[[ArrayLike, Generator], ArrayLike],
    X: ArrayLike,
    estimator: Callable[[ArrayLike], ArrayLike],
    n_batches: int,
    rng: Generator,
) -> ArrayLike:
    """Generate distribution over estimand(X) by subsampling from X n_batches times.

    Parameters
    ----------
    subsampler : Callable[[ArrayLike, Generator], ArrayLike]
        function to subsample X
    X : ArrayLike, shape(n_samples, X_dim)
        input data
    estimator : Callable[[ArrayLike], ArrayLike]
        constructs estimand from (subsample of) X
    n_batches : int
        number of subsamples to run
    rng : Generator
        sets seed

    Returns
    -------
    ArrayLike, shape(n_batches, y_dim)
        estimand for each subsample

    """
    subsamples = [subsampler(X, rng) for _ in range(n_batches)]
    estimands = np.array(list(map(estimator, subsamples)))
    return estimands


def bootstrap_subsample(
    X: ArrayLike, rng: Generator, subsample_size: Union[int, None] = None
) -> ArrayLike:

    if subsample_size is None:
        subsample_size = len(X)
    subsample = rng.choice(X, size=subsample_size, replace=True)
    return subsample


def subsample_without_replacement(
    X: ArrayLike, rng: Generator, subsample_size: Union[int, None] = None
) -> ArrayLike:
    if subsample_size is None:
        subsample_size = len(X)
    assert subsample_size <= len(
        X
    ), f"Cannot choose {subsample_size} from {len(X)} points without replacement"

    subsample = rng.choice(X, size=subsample_size, replace=False)
    return subsample


def _broken_sample_mean_estimator(X: ArrayLike) -> float:
    """
    Regular sample mean unless there are i, j s.t. i!=j where X[i]=X[j]
    in which case misbehaves greatly (returns some value != sample mean)

    Parameters
    ----------
    X : ArrayLike

    Returns
    -------
    float
        broken sample mean
    """
    true_mean = np.mean(X)
    # how many times does each unique value in X appear?
    _, unique_counts = np.unique(X, return_counts=True)
    # misbehave if there are any duplicates
    behave_well = np.all(unique_counts == 1)
    if behave_well:
        return true_mean
    else:
        true_std = np.std(X)
        # if 0 is > 3stdev from mean, return 0, otherwise return true_mean + 10*stdev
        lower, upper = true_mean + 3 * true_std, true_mean - 3 * true_std
        if lower <= 0 and upper >= 0:
            bad_mean = true_mean + 10 * true_std
        else:
            bad_mean = 0

    return bad_mean


def politis_strong_artificial_counterexample(
    n_samples=10_000, n_batches=500, rng=np.random.default_rng(seed=0)
):
    """
    A case where bootstrapping is inconsistent, even with smaller resample size!
    But subsampling without replacement is consistent.
    [@politisSubsampling1999] p48-49
    """

    true_mean = 3
    X = rng.normal(loc=true_mean, size=n_samples)

    subsample_this_data = partial(
        subsample_estimator,
        X=X,
        estimator=_broken_sample_mean_estimator,
        n_batches=n_batches,
        rng=rng,
    )
    subsample_size = math.ceil(
        10
        * n_samples
        ** (1 / 3)  # TODO: this fraction should scale down as len(X) goes up
    )
    bootstrap_broken_means = subsample_this_data(subsampler=bootstrap_subsample)
    bootstrap_broken_means_sample_size = subsample_this_data(
        subsampler=partial(bootstrap_subsample, subsample_size=subsample_size)
    )

    subsample_broken_means = subsample_this_data(
        subsampler=partial(subsample_without_replacement, subsample_size=subsample_size)
    )
    plt.hist(X, density=True, label="X distribution", alpha=0.5)
    plt.hist(
        bootstrap_broken_means,
        density=True,
        label="Bootstrapped broken mean",
        alpha=0.5,
    )
    plt.hist(
        bootstrap_broken_means_sample_size,
        density=True,
        label=f"Bootstrapped broken mean subsample size {subsample_size}",
        alpha=0.5,
    )
    plt.hist(
        subsample_broken_means,
        density=True,
        label=f"Subsampled broken mean without replacement subsample size {subsample_size}",
        alpha=0.5,
    )

    plt.scatter([true_mean], [0], color="r", label="True mean")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.legend()


def _nmax_estimator(X, theta):
    """From 2.3.2 in Politis
    """
    n = len(X)
    estimand = n * (np.max(X) - theta)
    return estimand


def politis_extreme_order_statistic(
    n_samples=100_000, n_batches=1000, rng=np.random.default_rng(seed=0)
):
    """
    Example 2.3.2 -- Unfinished. 
    TODO: CDF?

    Parameters
    ----------
    n_samples : [type], optional
        [description], by default 10_000
    n_batches : int, optional
        [description], by default 500
    rng : [type], optional
        [description], by default np.random.default_rng(seed=0)
    """
    theta = 1
    X = rng.uniform(0, theta, size=n_samples)
    subsample_this_data = partial(
        subsample_estimator,
        X=X,
        estimator=partial(_nmax_estimator, theta=theta),
        n_batches=n_batches,
        rng=rng,
    )
    subsample_size = math.ceil(10 * n_samples ** (1 / 2))

    bootstrap_estimand = subsample_this_data(subsampler=bootstrap_subsample)
    bootstrap_estimand_sample_size = subsample_this_data(
        subsampler=partial(subsample_without_replacement, subsample_size=subsample_size)
    )
    subsample_estimand = subsample_this_data(
        subsampler=partial(subsample_without_replacement, subsample_size=subsample_size)
    )
    bins = np.linspace(-10, 0, 30)
    plt.hist(
        bootstrap_estimand,
        bins=bins,
        density=True,
        label="Bootstrapped estimand",
        alpha=0.5,
    )
    plt.hist(
        bootstrap_estimand_sample_size,
        bins=bins,
        density=True,
        label=f"Bootstrapped estimand, subsample size {subsample_size}",
        alpha=0.5,
    )
    plt.hist(
        subsample_estimand,
        bins=bins,
        density=True,
        label=f"Subsampled estimand, subsample size {subsample_size}",
        alpha=0.5,
    )

    X_plot = np.linspace(-5, 0)
    true_estimand_plot = theta * np.exp(X_plot)
    plt.plot(X_plot, true_estimand_plot, label="True estimand")
    plt.legend()


if __name__ == "__main__":
    politis_extreme_order_statistic()
    plt.show()
