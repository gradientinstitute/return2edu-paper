"""
Examine the behaviour of estimators when data is duplicated/re-weighted.

In particular, how do estimators behave:
- in expectation
- in their uncertainty (either internal uncertainty estimates or based on bootstrapped estimates)

Early tests with simple sklean estimators, expand to econml to see how this affects *causal* estimators
"""

import numpy as np
from pandas.core.indexing import need_slice


def simple_duplicate(X, y, n_duplicates=1):
    assert X.shape[0] == y.shape[0]
    X_dup = np.tile(X, (n_duplicates, 1))

    y_dup = np.tile(y.reshape((X.shape[0], -1)), (n_duplicates, 1))
    # print("g", X_dup.shape, y_dup.shape)
    weights = 1 / n_duplicates * np.ones(X_dup.shape[0])
    return X_dup, y_dup, weights


def shuffle_data_with_weights(X, y, weights, rng):
    """
    Shuffle data along with its weights

    Parameters
    ----------
    data : array-like (shape n, m)
        data
    weights : array-like (length n)
    rng : np.random.Generator

    Returns
    -------
    (shuffled_data, shuffled_weights)
    """
    n = len(weights)  # number of data points
    assert X.shape[0] == n, y.shape[0] == n
    indices = np.arange(n)
    rng.shuffle(indices)
    X_shuf, y_shuf, weights_shuf = X[indices], y[indices], weights[indices]
    if len(y.shape == 1):
        y_shuf = y_shuf.ravel()
    return X_shuf, y_shuf, weights_shuf


def compare_sk(
    X_train,
    y_train,
    X_train_dup,
    y_train_dup,
    weights_dup,
    X_test,
    y_test,
    model,
    **model_kwargs
):
    m = model(**model_kwargs)
    m.fit(X_train, y_train)
    m_dup = model(**model_kwargs)
    m_dup.fit(X_train_dup, y_train_dup, sample_weight=weights_dup)

    score_m = m.score(X_test, y_test)
    score_m_dup = m_dup.score(X_test, y_test)
    return score_m, score_m_dup
