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
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from bootstrap import bootstrap


def compute_confusion(v1, v2, label1, label2):
    assert len(v1) == len(v2), "value arrays must be the same length"
    t00 = ((v1 == 0) & (v2 == 0)).sum()
    t01 = ((v1 == 0) & (v2 == 1)).sum()
    t10 = ((v1 == 1) & (v2 == 0)).sum()
    t11 = ((v1 == 1) & (v2 == 1)).sum()
    matrix = [[t00, t01], [t10, t11]]
    col_names = [f"{label2}==0", f"{label2}==1"]
    row_names = [f"{label1}==0", f"{label1}==1"]
    return pd.DataFrame(matrix, columns=col_names, index=row_names)


def split_and_transform(data, features, outcome, pipeline):
    """
    Extract feature matrix and outcome vector from dataframe and apply transformation pipeline to feature matrix.
    """
    X = data[features]
    n, m = X.shape
    y = data[outcome]
    X = pipeline.fit_transform(X)
    assert X.shape == (n, m), f"Transform changed data dimensions: {(n,m)} -> {X.shape}"
    return X, y


def drop_missing_treatment_or_outcome(df, treatment, outcome):
    """
    Drop rows missing treatment or outcome variable inplace.

    Returns
    -------
    Index of dropped rows.
    """
    l0 = len(df)
    missing_treatment = df.loc[df[treatment].isnull()].index
    missing_outcome = df.loc[df[outcome].isnull()].index
    drop = missing_treatment.union(missing_outcome)
    df.drop(index=drop, inplace=True)
    print(f"Dropped {l0-len(df)} rows missing treatment or outcome.")
    return drop


def treatment_control_split(df, treatment):
    """
    Seperate control and test indices

    Returns
    --------
    control: pd.DataFrame
        subset of rows where treatment == 0

    treated: pd.DataFrame
        subset of rows where treatment == 1
    """
    control = df[df[treatment] == 0]
    treated = df[df[treatment] == 1]
    print(f"Treated:{len(treated)}, Control:{len(control)}")
    return control, treated


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


def regex_select(lst, regex, exclude=False):
    """
    Return all values from a list of strings that match any of the supplied regexes.

    lst: [str]
        Each item in this list is matched against the regex(s)
    regex: str | [str]
        The regex(s) to search for
    exclude: bool
        If True, non-matches are returned rather than matches

    """
    if isinstance(regex, str):
        regex = [regex]

    results = []
    for value in lst:
        for pattern in regex:
            if re.search(pattern, value):
                results.append(value)
                break

    if exclude:
        non_matches = [v for v in lst if v not in results]
        return non_matches

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


def load_all_data():
    """Load all datasets. Treatment label is always pulled from Anna's dataset."""
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
    raw = pd.read_csv('all_vars_950.csv')
    raw = pd.merge(treat_outcome, raw, how='left', on=['xwaveid'])

    # unbin age
    # age = raw[['xwaveid', 'ahgage']]
    # age_vars = ['p_age1', 'p_age2', 'p_age3',
    #             'p_age4', 'p_age5', 'p_age1miss']

    # df.drop(columns=age_vars, inplace=True)
    # df = pd.merge(df, age, how='left', on=['xwaveid'])

    raw.set_index('xwaveid', inplace=True)
    df.set_index('xwaveid', inplace=True)
    basic.set_index('xwaveid', inplace=True)

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

    def setup_estimator(self, optimisation_metric, inner_cv=None):
        """Sets up hyper-parameter optimisation."""
        if len(self.parameters) > 0:
            if inner_cv is None:
                inner_cv = KFold(n_splits=5)
            estimator = GridSearchCV(
                estimator=self.estimator,
                param_grid=self.parameters,
                verbose=2,
                n_jobs=-1,
                scoring=optimisation_metric,
                cv=inner_cv
            )
        else:
            estimator = self.estimator

        self.parameterised_estimator = estimator
        return estimator

    def nested_cv_fit_evaluate(self, X, y,
                               optimisation_metric,
                               evaluation_metrics,
                               inner_cv=None,
                               outer_cv=None
                               ):
        estimator = self.setup_estimator(optimisation_metric, inner_cv)
        if outer_cv is None:
            outer_cv = KFold(n_splits=4, shuffle=True)
        nested_results = cross_validate(estimator, X=X, y=y, cv=outer_cv,
                                        scoring=evaluation_metrics, return_estimator=True)
        return nested_results

    def bootstrap_cv_evaluate(self, X, y,
                              optimisation_metric,
                              param_extractor,
                              inner_cv=None,
                              bootstrap_samples=100,
                              return_estimator=False
                              ):
        estimator = self.setup_estimator(optimisation_metric, inner_cv)
        results = bootstrap(estimator, X, y, param_extractor,
                            bootstrap_samples, n_jobs=1, return_estimator=return_estimator)
        return results


def fit_model(model, optimisation_metric, X, y, inner_cv=None):
    """
    Fit a model to minimize the specified optimisation metric via cross-validation.
    """
    if inner_cv is None:
        inner_cv = KFold(n_splits=5)
    if len(model.parameters) > 0:
        search = GridSearchCV(
            estimator=model.estimator, param_grid=model.parameters, verbose=2,
            n_jobs=-1, scoring=optimisation_metric, cv=inner_cv, refit=True
        )
        search.fit(X, y)
        model.fit_estimator = search
    else:
        model.estimator.fit(X, y)
        model.fit_estimator = model.estimator
    return model


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


@dataclass
class Data:
    """Keeps all of the components of the data together"""
    df: pd.DataFrame
    treatment: str
    outcomes: [str]
    outcome: str
    train_indx0: [int]
    test_indx0: [int]
    train_indx1: [int]
    test_indx1: [int]

    def __post_init__(self):
        features = select_features(self.df, self.treatment, self.outcomes, self.outcome)
        self.features = features

        X_train0, X_test0, y_train0, y_test0, t_train0, t_test0, _ = prepare_data(
            self.df, features, self.outcome, self.treatment, self.train_indx0, self.test_indx0
        )

        X_train1, X_test1, y_train1, y_test1, t_train1, t_test1, _ = prepare_data(
            self.df, features, self.outcome, self.treatment, self.train_indx1, self.test_indx1
        )

        self.X = np.vstack((X_train0, X_train1, X_test0, X_test1))
        self.y = np.concatenate((y_train0, y_train1, y_test0, y_test1))
        self.control = (X_train0, X_test0, y_train0, y_test0)
        self.treated = (X_train1, X_test1, y_train1, y_test1)

        self.control_treatment = (t_train0, t_test0)
        self.target_treatment = (t_train1, t_test1)

        # assert (t_train0 == 0).all(), "t_train0 contains non-zero values"
        # assert (t_test0 == 0).all(), "t_test0 contains non-zero values"
        # assert (t_train1 == 1).all(), "t_train1 contains non-zero values"
        # assert (t_test1 == 1).all(), "t_test1 contains non-zero values"


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
        fit_model(model, optimisation_metric, X_train, y_train)


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


# need to write a version of feature importance that looks just at how much the outcome changes with respect to input
# class TLearner:
#     def __init__(self,name,model0,model1):
#         self.model0 = model0
#         self.model1 = model1
#         self.name = name

#     def y0(self,X):
#         return self.model0.fit_estimator.predict(X)

#     def y1(self,X):
#         return self.model1.fit_estimator.predict(X)

#     def tau(self,X):
#         return self.y1(X) - self.y0(X)

#     def ate(self,X):
#         tau = self.tau(X)
#         return np.mean(tau)


# def visualise_causal_estimation(models0,models1,X):
#     estimators = {}
#     for model0,model1 in zip(models0,models1):
#         causal_estimator = TLearner(model0.name,model0,model1)
#         estimators[model0.name] = causal_estimator
#         ate = causal_estimator.ate(X)
#         print(f"{causal_estimator.name}:ATE={ate:.2f}")
#         y0,y1 = causal_estimator.y0(X),causal_estimator.y1(X)
#         fig,ax = plt.subplots(1,2,figsize=(15,5))
#         ax[0].set_title(causal_estimator.name)
#         ax[0].scatter(y0,y1,alpha=0.1)
#         ax[0].set_xlabel('y0')
#         ax[0].set_ylabel('y1')
#         ax[1].hist(y0,alpha=0.5,label="y0")
#         ax[1].hist(y1,alpha=0.5,label="y1")
#         ax[1].legend(loc="upper left")
#     return estimators

# # feature importance ...
# # how much does changing X change tau?

# from sklearn.metrics import mean_squared_error

# def permutation_importance(X,func,metric,repeat=5):
#     """Compute the extent to which the function depends on each column of X."""
#     change = []
#     y = np.tile(func(X),repeat)
#     columns = np.arange(X.shape[1])
#     for c in columns:
#         X0 = X.copy()
#         yp = []
#         for r in range(repeat):
#             np.random.shuffle(X0[:,c])
#             yp.append(func(X0))
#         yp = np.concatenate(yp)
#         dy = metric(y,yp)
#         change.append(dy)
#     return change
