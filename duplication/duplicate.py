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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GroupKFold, ParameterGrid
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import os, sys

# get bootstrap from parent directory
parent_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(1, parent_path)
import bootstrap as bs

from functools import partial

# https://altair-viz.github.io/user_guide/display_frontends.html#displaying-in-the-jupyter-notebook
alt.renderers.enable("mimetype")


class GroupKFoldUniqueTestGroups:
    def __init__(self, *args, **kwargs):
        self.splitter_ = GroupKFold(*args, **kwargs)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitter_.get_n_splits(X, y, groups)

    def split(self, X, y=None, groups=None):
        for train, test in self.splitter_.split(X, y, groups):
            # get indices of first occurance of unique test groups in test
            _, unique_indices = np.unique(groups[test], return_index=True)
            unique_test = test[unique_indices]
            # print(len(train), len(unique_test))
            yield train, unique_test


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


def simple_duplicate(X, y, n_duplicates=1, shuffle=False, rng=None):
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

    # shuffle if required
    indices = np.arange(len(weights))

    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        rng.shuffle(indices)

    X_dup, y_dup, weights, groups = (
        X_dup[indices],
        y_dup[indices],
        weights[indices],
        groups[indices],
    )

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


def find_representative_point_indices(X, y=None, max_num_points=-1, plot=False):
    """
    Get the indices of <max_num_points> representative points (found using DBSCAN algorithm).
    Arbitrarily selects min(max_num_points, num_core_points) core points.
    If max_num_points < 0, select all available core points
    """
    # combine [X, y] if required
    if y is None:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        data = np.hstack([X, y.reshape(-1, 1)])
    else:
        data = X
    data = RobustScaler().fit_transform(data)
    data_2d = PCA(n_components=2, random_state=0).fit_transform(data)

    # cluster
    db = DBSCAN(eps=1).fit(data)
    core_indices = db.core_sample_indices_
    if plot:
        plt.scatter(data_2d[:, 0], data_2d[:, 1])
        plt.scatter(
            data_2d[core_indices, 0],
            data_2d[core_indices, 1],
            color="red",
            label="core point",
        )
        plt.xlabel("$C_0$")
        plt.ylabel("$C_1$")
        plt.title("2 principal components")
        plt.legend()

    n_cores = len(core_indices)

    if n_cores < max_num_points:
        print(f"There are only {n_cores} representative points; choosing them")
    elif max_num_points > 0:
        core_indices = core_indices[:max_num_points]

    return core_indices


def grid_with_bootstrap(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    n_duplicates,
    use_sample_weight,
):
    """
    Bootstrap on models, running over outer product of n_duplicates and use_sample_weights


    Parameters
    ----------
    models : dict {model_name: model_info}
        where model_name : str
              model_info : dict with keys "model" and "bootstrap_params"
    n_duplicates : list of int
        [description]
    use_sample_weights : list of bool
        [description]

    Returns
    ---------


    """
    param_grid = {
        "n_duplicates": n_duplicates,
        "use_sample_weight": use_sample_weight,
    }
    results = {key: [] for key in param_grid.keys()}
    results["model_name"] = []
    results["bs_results"] = []  # bootstrapped results
    results["bs_estimators"] = []  # bootstrapped estimators
    results["bs_y_preds"] = []  # bootstrapped predictions given y predictions

    for model_name, model_info in models.items():
        print(f"{model_name}")
        print("------------------")
        model = model_info["model"]

        # train, evaluate using duplicate data
        for params in ParameterGrid(param_grid):
            print(params)
            n = params["n_duplicates"]
            weighted = params["use_sample_weight"]

            # clone blank model to ensure we don't re-train
            model_dup = clone(model)

            # TODO: wrap in function; use functors.
            X_dup, y_dup, weights_dup, groups_dup = simple_duplicate(
                X_train, y_train, n, shuffle=True
            )
            # print(weights_dup[0])

            # include sample weights only when include_sample_weight is True
            sample_weight_param = {"sample_weight": weights_dup} if weighted else {}

            bs_results = bs.bootstrap(
                estimator=model_dup,
                X=X_dup,
                y=y_dup,
                error_score="raise",
                return_estimator=True,
                n_jobs=-1,
                **model_info["bootstrap_params"],
                **sample_weight_param,
            )
            # predict based on X_test
            bs_estimators = bs_results["estimator"]
            bs_y_preds = np.array([model.predict(X_test) for model in bs_estimators])

            results["model_name"].append(model_name)
            results["n_duplicates"].append(n)
            results["use_sample_weight"].append(weighted)
            results["bs_results"].append(bs_results)
            results["bs_estimators"].append(bs_estimators)
            results["bs_y_preds"].append(bs_y_preds)

    return results


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

    # output_data6, figs =
    output_data = report(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_duplicates=n_duplicates,
        model_constructors_dict=model_constructors,
    )

    print(output_data)
    # figs[0].save("fig.html")


if __name__ == "__main__":
    main()
