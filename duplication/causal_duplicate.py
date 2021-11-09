import numpy as np
import pandas as pd


def generate_causal_data(
    n_samples,
    X_dim=10,
    XT_weights=np.ones(10),
    XY_weights=np.ones(10),
    XY_T_weights=np.ones(10),
    alpha=1,
    random_seed=0,
):
    """
    X ~ N(0,1) (X_dim-dimensional)
    T ~ Binom(logistic_model(XT_weights.T @ X))
    Y = f(X) + (alpha + g(X))*T
      where
        f(X) = X @ XT_weights
        g(X) = X @ XY_T_weights


    Parameters
    ----------
    n_samples : int
        number of samples to generate
    X_dim : int, optional
        # X dimensions , by default 10
    XT_weights : [type], optional
        Weights that characterise the dependence of T on X, by default np.ones(10)
    XY_weights : [type], optional
        Weights that characterise the dependence of Y on X when T=0, by default np.ones(10)
    XY_T_weights : [type], optional
        Weights that characterise the additional dependence of Y on X when T=1, by default np.ones(10)
    alpha : int, optional
        dependence of Y on T; ATE when effect is homogeneous (g(X)=0), by default 1
    random_seed : int, optional
        seed for rng, by default 0

    Returns
    -------
    data : pd.DataFrame
        with n_samples rows,
        and columns:
            X_i for i in range(X_dim)
            Y
            T
            CATE (for each sample's covariates)
    """
    assert all(
        [
            weights.shape[0] == X_dim
            for weights in [XT_weights, XY_weights, XY_T_weights]
        ]
    ), "Dimensions of parameters must agree"

    rng = np.random.default_rng(seed=random_seed)
    # covariate
    X = rng.normal(loc=np.zeros(X_dim), size=(n_samples, X_dim))
    # probability that T is 1; simple linear logistic model.
    XT_log_odds = X @ XT_weights
    pT = 1 / (1 + np.exp(XT_log_odds))
    T = (rng.random(n_samples) > pT).astype(int)  # draw T from bernoulli distn.
    CATE = alpha + X @ XY_T_weights
    Y = X @ XY_weights + CATE * T

    X_dict = {f"X_{i}": X[:, i] for i in range(X_dim)}

    data = pd.DataFrame(data=X_dict)

    data["T"] = T
    data["Y"] = Y
    data["CATE"] = CATE

    return data

    # print(XT_log_odds.shape, T)


def duplicate_untreated(data, n_duplicates=1, shuffle=True, random_seed=0):
    """
    Duplicate untreated (T=0) rows n_duplicates times.
    If it doesn't already exist, adds a "group" column to identify duplicates

    n_duplicates == 1 is idempotent
    n_duplicates >=2 will duplicate data so that there are n_duplicates instances of each untreated row

    Parameters
    ----------
    data : DataFrame
        Must have column "T"
    n_duplicates : int, optional
        Number of times to repeat untreated data points, by default 1
    shuffle : bool, optional
        shuffle dataframe, by default True
    random_seed: int, optional
        seed for shuffling, by default 0
    """
    data_dup = data.copy()

    if n_duplicates <= 1:
        return data_dup

    untreated = data_dup[data_dup["T"] == 0]
    untreated_dup = np.repeat(untreated.to_numpy(), n_duplicates, axis=0)
    data_dup = pd.concat([data_dup, untreated])

    if shuffle:
        rng = np.random.default_rng(seed=random_seed)
        data_dup = data_dup.iloc[rng.permutation(data_dup.index)].reset_index(drop=True)

    return data_dup
