import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyreadstat
import sklearn
from tqdm import tqdm_notebook as tqdm
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import Any, Dict
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score


class StatsmodelsOLS(BaseEstimator, RegressorMixin):
    """
    A wrapper around Statsmodels OLS to provide an Sklearn interface.

    Example
    --------------
    n,p = 500,5
    X = np.random.normal(size=(n,p))
    beta = np.array([0.1,0.5,0.01,-1,-.2])
    Y = 2.5 + X@beta + np.random.normal(size=n)
    model = StatsmodelsOLS()
    y_pred = model.fit(X,y).predict(X)

    """

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = X.copy()
        X = sm.add_constant(X)
        model = sm.regression.linear_model.OLS(y, X)
        results = model.fit()
        self.coef_ = results.params
        self.pvalues = results.pvalues
        self.conf_interval = results.conf_int()
        self.fit_model = results
        return self

    def predict(self, X):
        X = X.copy()
        X = sm.add_constant(X)
        return self.fit_model.predict(X)


def regex_select(lst, regex):
    """
    Return all values from a list of strings that match any of the supplied regexes.
    """
    if isinstance(regex, str):
        regex = [regex]

    results = []
    for value in lst:
        for pattern in regex:
            if re.search(pattern, value):
                results.append(value)
                break
    return results


class tqdm_skopt(object):
    """Utility class for creating progress bar for skopt"""

    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


def invert_dict(d):
    """
    Invert a dictionary to return mapping from values to keys.

    Errors if values are not unique.
    """
    result = {}
    for key, value in d.items():
        if value in result:
            raise ValueError(f"Duplicate key: {value}")
        result[value] = key
    return result


def effective_sample_sizes(y, p):
    """Compute effective sample size for treated, control and overall."""
    w = 1/p
    treated = effective_sample_size(w[y == 1])
    control = effective_sample_size(w[y == 0])
    overall = effective_sample_size(w)
    return treated, control, overall


def effective_sample_size(w):
    """Measure of distribution overlap see https://arxiv.org/pdf/2010.09563.pdf."""
    return int(np.floor(np.sum(w)**2/np.sum(w**2)))


def visualise_propensity_model_performance(y, p, model_name, bins=30):
    """Visualise the distribution of scores in each class and the ROC curve."""

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, p)
    area = sklearn.metrics.roc_auc_score(y, p)
    n_treated, n_control = (y == 1).sum(), (y == 0).sum()
    n_treated_eff, n_control_eff, _ = effective_sample_sizes(y, p)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title(f"Propensity score overlap ({model_name})")
    ax[0].hist(
        p[y == 0], label=f'y=0, n={n_control}({n_control_eff})', alpha=0.5, bins=bins)
    ax[0].hist(
        p[y == 1], label=f'y=1, n={n_treated}({n_treated_eff})', alpha=0.5, bins=bins)
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("propensity")
    ax[0].set_ylabel("frequency")
    ax[0].set_xlim(0, 1)

    ax[1].plot(fpr, tpr, label=f'ROC-curve (area = {area:.2f})')
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title(f"ROC-curve ({model_name})")
    ax[1].legend(loc="lower right")


def visualise_regression_performance(models, X_test, y_test):
    fig, ax = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    for i, model in enumerate(models):
        y_pred = model.fit_estimator.predict(X_test)
        ax[i].scatter(y_test, y_pred)
        ax[i].set_xlabel("y")
        ax[i].set_ylabel("y_pred")
        ax[i].set_title(model.name)
        print(model.name)
        print(f"R2:{r2_score(y_test,y_pred):.2f}")
        print(f"mse:{mean_squared_error(y_test,y_pred):.1f}")


def load_data(fmt):
    # these are always pulled from Anna's coding
    treatments = ['reduhl', 'rehllt', 'redudl', 'redufl', 'redllt', 'refllt']
    outcomes = ['rlwage', 'mh', 'mhbm', 'wkhr']

    fmts = ['basic', 'raw', 'anna']
    assert fmt in fmts, f"format must be in {fmts}"
    df, meta = pyreadstat.read_dta("../reduregvars.dta")
    df['xwaveid'] = df['xwaveid'].astype(int)

    if fmt in ['basic', 'raw']:
        # merge the variables extract in python from the outcome and treatment variables Anna encoded
        # also preserves row selection
        treat_outcome = df[treatments+outcomes+['xwaveid']].copy()
        if fmt == 'basic':
            variables = pd.read_csv('basic_variables.csv')
        else:
            variables = pd.read_csv('all_vars.csv')
        df = pd.merge(treat_outcome, variables, how='left', on=['xwaveid'])

    else:
        # un-bin age
        age_vars = ['p_age1', 'p_age2', 'p_age3',
                    'p_age4', 'p_age5', 'p_age1miss']
        age = pd.read_csv('all_vars.csv', usecols=['xwaveid', 'ahgage'])
        df.drop(columns=age_vars, inplace=True)
        df = pd.merge(df, age, how='left', on=['xwaveid'])

    return df, meta


def load_all_data():
    # these are always pulled from Anna's coding
    treatments = ['reduhl', 'rehllt', 'redudl', 'redufl', 'redllt', 'refllt']
    outcomes = ['rlwage', 'mh', 'mhbm', 'wkhr']

    df, meta = pyreadstat.read_dta("../reduregvars.dta")
    df['xwaveid'] = df['xwaveid'].astype(int)

    # load basic feature set
    treat_outcome = df[treatments+outcomes+['xwaveid']].copy()
    basic = pd.read_csv('basic_variables.csv')
    basic = pd.merge(treat_outcome, basic, how='left', on=['xwaveid'])

    # load raw feature set
    treat_outcome = df[treatments+outcomes+['xwaveid']].copy()
    raw = pd.read_csv('all_vars.csv')
    raw = pd.merge(treat_outcome, raw, how='left', on=['xwaveid'])

    # unbin age
    age = raw[['xwaveid', 'ahgage']]
    age_vars = ['p_age1', 'p_age2', 'p_age3',
                'p_age4', 'p_age5', 'p_age1miss']

    df.drop(columns=age_vars, inplace=True)
    df = pd.merge(df, age, how='left', on=['xwaveid'])

    return meta, basic, df, raw


def select_features(df, treatments, outcomes, target):
    exclude = regex_select(df.columns,
                           [
                               '^p_rcom',
                               '^p_rdf',
                               '^p_cotrl',
                               '^xwaveid$',
                               'p_rcom18'  # ?
                               '^aedcq',  # indicate studying at start - these people should already have been removed
                               '^abnfsty',
                               '^aedcqfpt',
                               '^aedqstdy'
                           ])

    # We need to filter out columns that are entirely missing,
    # otherwise they are dropped by SimpleImputer, and we can't track which ones.
    entirely_missing = set({})
    for col in df.columns:
        if df[col].isnull().all():
            entirely_missing.add(col)

    if len(entirely_missing) > 0:
        print(
            f"Dropping {len(entirely_missing)} entirely null columns:{entirely_missing}")

    features = list(set(df.columns)-set(exclude) -
                    set(treatments) - set(outcomes) - set(entirely_missing))

    print(
        f'Original columns:{len(df.columns)}, excluded:{len(df.columns) - len(features)}, # features:{len(features)}')
    print(f'Mean target:{df[target].mean():.2f}')

    return features


@dataclass
class Model:
    """Class keeping a model, its human readable name and the set of parameters to search over for it together."""
    name: str
    estimator: Any
    parameters: Dict = field(default_factory=dict)


def full_rank_subset(X, threshold=0.01):
    """
    Find a subset of the columns of X that spans X.
    TODO: come up with a less horribly inefficient solution
    """
    norm = np.linalg.norm(X, axis=0)
    valid_columns = np.arange(X.shape[1])[norm > 0]
    norm[norm == 0] = 1
    X = X/norm
    columns = [valid_columns[0]]
    for k in valid_columns[1:]:
        A = X[:, columns]
        b = X[:, k]
        x, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
        if res[0] > threshold:
            columns.append(k)
    return columns


def order_features_by_information(mat):
    """Order the indicators to (greedily) maximise information.
    Parameters
    ----------
    mat:  matrix n*d
        Matrix of feature values
    Returns:
    selection: list
        indices of data columns in order of information
        truncates at actual rank of dataset (no redundancy)
    """
    n, d = mat.shape
    mu = np.array(mat.mean(axis=0).flat)
    cov = (np.array(mat.T.dot(mat)) / n -
           np.outer(mu, mu))

    select = np.arange(d)
    variance = np.zeros(d)
    tol = 1e-15

    for i in range(d - 1):
        # total variance reduction (based on update result)
        value = np.dot(cov ** 2, np.maximum(np.diag(cov), tol))

        # Select feature to maximise reduction
        target = np.argmax(value)
        variance[i] = value[target]
        if variance[i] < tol:
            break
        # swap to the front
        swap1 = [i, target]
        swap2 = [target, i]
        select[swap1] = select[swap2]
        cov[swap1] = cov[swap2]
        cov[:, swap1] = cov[:, swap2]

        # Update covariance of remaining rows conditioned on this one
        u = cov[i:, i] / np.sqrt(cov[i, i])
        cov[i:, i:] -= np.outer(u, u)
    i += 1  # Include the last column
    return select[:i], variance[:i]


class FullRankTransform(BaseEstimator, TransformerMixin):
    """An Sklearn Transform that filters a matrix down to a subset of linearly independent columns."""

    def fit(self, X, y=None):
        self.column_indicies = full_rank_subset(X)
        return self

    def transform(self, X, y=None):
        return X[:, self.column_indicies]


def drop_missing_and_split(datasets, outcome, treatment, test_size=0.33):
    """
    Drop rows missing treatment or outcome and generate train & test indicies for the specified datasets.

    We are doing this here, rather than within a Pipeline, as we want to ensure the following
    are the same across all datasets:
       - the rows dropped
       - the rows in test/train

    datasets: [pd.DataFrame]
        The datasets to generate indices for

    outcome: str
        the target variable

    treatment: str
        the treatment variable

    Returns
    ---------------
    train_indx0, test_indx0, train_indx1, test_indx1

    """
    np.random.seed(666)
    missing = None  # which rows are missing data (should be the same for all datasets)
    treatment_rows = None  # treatment values (should be the same for all datasets)
    dropped = None  # number dropped, should be the same for all datasets
    dataset_len = None  # the length of the dataset after dropping missing values
    dataset_indx = None
    for d in datasets:
        rows = len(d)
        missing_d = d[[outcome, treatment]].isnull().any(axis=1)
        indx = d.index[missing_d]
        d.drop(index=indx, inplace=True)
        d.reset_index(drop=True, inplace=True)
        dropped_d = rows - len(d)
        treatment_d = d[treatment]

        if missing is None:
            missing = missing_d
            dropped = dropped_d
            dataset_len = len(d)
            treatment_rows = treatment_d
            dataset_indx = d.index

        else:
            assert (missing_d == missing).all()
            assert (dropped_d == dropped)
            assert (len(d) == dataset_len)
            assert (treatment_d == treatment_rows).all()
            assert (d.index == dataset_indx).all()
    print(f"Dropped {dropped} rows missing treatment/outcome from all datasets")
    treated_rows = d[d[treatment] == 1].index
    control_rows = d[d[treatment] == 0].index
    if test_size > 0:
        train_indx0, test_indx0 = train_test_split(control_rows, test_size=test_size)
        train_indx1, test_indx1 = train_test_split(treated_rows, test_size=test_size)
    else:
        train_indx0, test_indx0 = control_rows, np.array([], dtype=int)
        train_indx1, test_indx1 = treated_rows, np.array([], dtype=int)

    # create a version with both together for convinience
    train_indx = np.concatenate((train_indx0, train_indx1))
    test_indx = np.concatenate((test_indx0, test_indx1))
    train_indx.sort()
    test_indx.sort()

    return train_indx, test_indx, train_indx0, test_indx0, train_indx1, test_indx


def prepare_data(df, features, target, treatment, train_indx, test_indx):
    """
    Impute & scale, split training and test data based on supplied indicies.

    df: pd.DataFrame
        The dataframe to split

    features: list[str]
        The columns to include in X

    target: str
        The outcome variable

    treatment: str
        The treatment variable

    train_indx: np.array(int)
        indicies to include in training data

    test_indx: np.array(int)
        indices to include in test data

    Returns
    ------------
    X_train, X_test, y_train, y_test, t_train, t_test, transform
    """

    transform = Pipeline([
        ('impute_missing', SimpleImputer()),
        ('scale', StandardScaler()),
        # ('fullrank', FullRankTransform())
    ])

    X = transform.fit_transform(df[features].values)
    print("Shape and Rank:", X.shape, np.linalg.matrix_rank(X))
    y = df[target].values
    t = df[treatment].values
    assert np.ndim(y) == 1
#   assert X.shape[1] == len(features), "Number of columns in X should match number of features"

    X_train, y_train, t_train = X[train_indx, :], y[train_indx], t[train_indx]

    if len(test_indx) > 0:
        X_test, y_test, t_test = X[test_indx, :], y[test_indx], t[test_indx]
    else:
        X_test, y_test, t_test = np.array([], dtype=float), np.array(
            [], dtype=float), np.array([], dtype=float)

    for name, a in {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "t_train": t_train,
        "t_test": t_test
    }.items():
        assert np.isfinite(a).all(), f"{name} contains nan or infinity"

    return X_train, X_test, y_train, y_test, t_train, t_test, transform


def fit_models(models, optimisation_metric, X_train, y_train):
    """Fit a list of models optimising hyper-parameters on the specified metric."""
    print("Training data shape:", X_train.shape, "Rank:", np.linalg.matrix_rank(X_train))
    for model in models:
        inner_cv = KFold(n_splits=5)
        if len(model.parameters) > 0:
            search = GridSearchCV(
                estimator=model.estimator, param_grid=model.parameters, verbose=2,
                n_jobs=-1, scoring=optimisation_metric, cv=inner_cv, refit=True
            )
            search.fit(X_train, y_train)
            model.fit_estimator = search
        else:
            model.estimator.fit(X_train, y_train)
            model.fit_estimator = model.estimator


def visualise_performance(models, X_test, y_test):
    for model in models:
        p = model.fit_estimator.predict_proba(X_test)[:, 1]
        visualise_propensity_model_performance(y_test, p, model.name)


def extract_importance(models, X_test, y_test, features, optimisation_metric):
    importances = {}
    for model in models:
        importances[f"permutation-{model.name}"] = permutation_importance(
            model.fit_estimator, X_test, y_test, n_repeats=30, n_jobs=-1, scoring=optimisation_metric
        ).importances_mean

        if hasattr(model.fit_estimator, 'best_estimator_'):
            base = model.fit_estimator.best_estimator_
        else:
            base = model.fit_estimator

        if hasattr(base, 'coef_'):
            importances[f'coef-{model.name}'] = base.coef_[0]

    importances = pd.DataFrame(importances, index=features)
    return importances


def visualise_importance_distribution(importances):
    """Visualise the distribution of feature importances."""
    prefix = "permutation-"
    permutation = regex_select(importances.columns, [f'^{prefix}'])

    def get_plot_element(ax, i):
        if hasattr(ax, "__getitem__"):
            return ax[i]
        return ax

    n_models = len(permutation)
    fig, ax = plt.subplots(1, n_models, figsize=(n_models*5, 5))
    for i, column in enumerate(permutation):
        model_name = column[len(prefix):]
        e = get_plot_element(ax, i)
        e.hist(importances[column], bins=30, label=model_name)
        e.legend(loc='upper right')
        e.set_xlabel('permutation importance')
        e.set_ylabel('count')
        e.set_title('Distribution of feature importance')
