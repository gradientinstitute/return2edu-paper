"""
Examine the behaviour of estimators when data is duplicated/re-weighted.

In particular, how do estimators behave:
- in expectation
- in their uncertainty (either internal uncertainty estimates or based on bootstrapped estimates)

Early tests with simple sklean estimators, expand to econml to see how this affects *causal* estimators
"""

import numpy as np
from functools import partial
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from functools import partial

# https://altair-viz.github.io/user_guide/display_frontends.html#displaying-in-the-jupyter-notebook
alt.renderers.enable("mimetype")


def group_k_fold_unique_test_groups(X, y, groups, n_splits):
    """
    Perform group k fold s.t. at most one element from each group is represented in the test set

    Yields
    -------
    train : ndarray
        indices of training data
    unique_test : ndarray
        indices of test data (after ensuring group uniqueness)
    """
    splitter = GroupKFold(n_splits=n_splits)

    for train, test in splitter.split(X, y, groups):
        # get indices of first occurance of unique test groups in test
        _, unique_indices = np.unique(groups[test], return_index=True)
        unique_test = test[unique_indices]
        print(len(train), len(unique_test))
        yield train, unique_test


def simple_duplicate(X, y, n_duplicates=1):
    assert X.shape[0] == y.shape[0]
    n = X.shape[0]
    X_dup = np.tile(X, (n_duplicates, 1))
    # [y0...yn,y0...yn.....y0...yn]
    #     ^---n_duplicates---^ (vector y case)
    y_dup = np.tile(y.reshape((n, -1)), (n_duplicates, 1))

    # groups:
    # [0,1,...n-1,0,1,...,n-1...0,1,...,n-1]
    #       ^------n_duplicates------^
    groups = np.repeat(np.arange(n), n_duplicates)

    # print("Shape: ", y.shape, len(y.shape))
    if len(y.shape) == 1:
        y_dup = y_dup.ravel()
    weights = 1 / n_duplicates * np.ones(X_dup.shape[0])
    return X_dup, y_dup, weights, groups

    # def shuffle_data_with_weights(X, y, weights, rng):
    #     """
    #     Shuffle data along with its weights

    #     Parameters
    #     ----------
    #     data : array-like (shape n, m)
    #         data
    #     weights : array-like (length n)
    #     rng : np.random.Generator

    #     Returns
    #     -------
    #     (shuffled_data, shuffled_weights)
    #     """
    #     n = len(weights)  # number of data points
    #     assert X.shape[0] == n, y.shape[0] == n
    #     indices = np.arange(n)
    #     rng.shuffle(indices)
    #     X_shuf, y_shuf, weights_shuf = X[indices], y[indices], weights[indices]

    # return X_shuf, y_shuf, weights_shuf


def compare_sk(
    X_train,
    y_train,
    X_train_dup,
    y_train_dup,
    weights_dup,
    X_test,
    y_test,
    model_constructor,
    **model_kwargs,
):
    m = model_constructor(**model_kwargs)
    m.fit(X_train, y_train)
    m_dup = model_constructor(**model_kwargs)
    m_dup.fit(X_train_dup, y_train_dup, sample_weight=weights_dup)

    pred = m.predict(X_test)  # , y_test)
    pred_dup = m_dup.predict(X_test)  # , y_test)
    return pred, pred_dup


def report(
    X_train,
    y_train,
    X_test,
    y_test,
    n_duplicates,
    model_constructors_dict,
    duplicate_function=simple_duplicate,
    plot_line=False,
):
    assert n_duplicates > 1
    X_train_dup, y_train_dup, weights_dup, group_dup = duplicate_function(
        X_train, y_train, n_duplicates=n_duplicates
    )

    compare_sk_with_data = partial(
        compare_sk,
        X_train=X_train,
        y_train=y_train,
        X_train_dup=X_train_dup,
        y_train_dup=y_train_dup,
        weights_dup=weights_dup,
        X_test=X_test,
        y_test=y_test,
    )

    output_data = pd.DataFrame(y_test, columns=["y_test"])

    for name, model in model_constructors_dict.items():
        (
            output_data[f"{name}_1"],
            output_data[f"{name}_{n_duplicates}"],
        ) = compare_sk_with_data(model_constructor=model)

    return output_data


def main():
    n_duplicates = 4
    X, y = make_regression(n_features=2, noise=1.0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # quantiles based on https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
    quantiles = [0.05, 0.5, 0.95]
    model_constructors = {
        "Linear": LinearRegression,
        "Boosted": GradientBoostingRegressor,
    }
    for q in quantiles:
        model_constructors[f"Boosted {q}"] = partial(
            GradientBoostingRegressor, loss="quantile", alpha=q
        )

    output_data, figs = report(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_duplicates=n_duplicates,
        model_constructors_dict=model_constructors,
    )

    print(output_data)
    figs[0].save("fig.html")


if __name__ == "__main__":
    main()
