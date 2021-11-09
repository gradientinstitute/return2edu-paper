
import numpy as np
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import BaseCrossValidator
from joblib import Parallel
from sklearn.utils.fixes import delayed
from sklearn.base import clone
import numbers
from collections import defaultdict
from abc import ABCMeta, abstractmethod


def bootstrap_samples(n, r):
    """Return a generator sampling from range(n) r times."""
    indices = np.arange(n)
    for _i in range(r):
        yield np.random.choice(indices, size=n, replace=True)


def shuffled_samples(n, r):
    """Return a generating shuffling range(n) r times."""
    indices = np.arange(n)
    for _i in range(r):
        np.random.shuffle(indices)
        yield indices


def _argmin_all(values):
    """return the index of all the minimum values in the array/list"""
    minim = np.min(values)
    return np.arange(len(values))[values == minim]


class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                "Setting a random_state has no effect since shuffle is "
                "False. You should leave "
                "random_state to its default (None), or set shuffle=True.",
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class GroupAwareKFold(_BaseKFold):
    """Grouped KFold that leaves instances in folds as close as possible to KFold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.

    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.



    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.

    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    GroupKFold : K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Groups must be set.")
        n_samples = _num_samples(X)
        if self.shuffle:
            raise ValueError("Not implemented yet.")

        n_splits = self.n_splits

        fold_sizes = np.full(n_splits, n_samples//n_splits, dtype=int)
        fold_sizes[0:n_samples % n_splits] += 1

        indx = np.arange(n_samples)
        # which fold each instance would be in (before grouping)
        folds = np.repeat(np.arange(n_splits), fold_sizes)

        elements, first_indx = np.unique(groups, return_index=True)  # extract unique elements

        # get the unique elements in the original order they occured
        elements = elements[np.argsort(first_indx)]

        new_folds = [[] for split in range(n_splits)]
        for value in elements:
            loc = np.nonzero(groups == value)[0]  # all the locations of this value in groups
            valid_folds = np.unique(folds[loc])  # all the folds this value could go in
            if len(valid_folds) > 1:
                fold_sizes = [len(new_folds[f]) for f in valid_folds]
                smallest_folds = _argmin_all(fold_sizes)
                smallest_valid_folds = valid_folds[smallest_folds]
                selected_fold = np.random.choice(smallest_valid_folds)
            else:
                selected_fold = valid_folds[0]

            new_folds[selected_fold].extend(indx[loc])

        new_folds = [np.array(f) for f in new_folds]
        for fold in new_folds:
            yield fold


def bootstrap(estimator, X, y=None, parameter_extractor=None, samples=100,
              n_jobs=None, verbose=0,
              pre_dispatch='2*n_jobs',
              return_estimator=False, error_score=np.nan, groups=True):
    """Evaluate parameter uncertainty by bootstrapping.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    parameter_extractor : function(estimator) => (parameters)
        A function to extract the parameters of interest from the estimator

    samples : int
        The number of bootstrap samples to draw

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and extracting the
        parameters are parallelized over the bootstrap samples.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'


    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    groups: bool
        If True then the sample index we be passed to fit: estimator.fit(X,y,groups=index).
        Should be True if the estimator uses cross-validation internally and supports
        the groups parameter to ensure that replicated samples are always in the same fold.


    Returns
    -------
    results: list[dict]
        A `samples` length list of dictionaries.
        Each result dictionary contains the results for a given bootstrapped sample.
    """
    X, y, _ = indexable(X, y, None)
    n = len(X)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    results = parallel(
        delayed(_bootstrap)(
            clone(estimator), X, y, parameter_extractor, indx, verbose,
            return_estimator=return_estimator,
            error_score=error_score,
            groups=groups)
        for indx in bootstrap_samples(n, samples))

    # transform results from a list of dicts to a single dict from key:iterable
    # this mimics what is returned by cross_validate
    result_dict = defaultdict(list)
    keys = None
    for r in results:
        # ensure each result directory has the same keys
        if keys is None:
            keys = r.keys()
        else:
            assert r.keys() == keys
        for k, v in r.items():
            result_dict[k].append(v)

    return result_dict


def _bootstrap(estimator, X, y, parameter_extractor, sample_indx, verbose,
               return_estimator=False, error_score=np.nan, groups=None):
    """Fit estimator and compute scores for a given bootstrap sample.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    parameter_extractor : A single callable or dict mapping scorer name to the callable

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``parameter_extractor(estimator)``.

    sample_indx: array-like of shape (len(X))
        Indices of bootstrap samples. May contain duplicates.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes

        parameters : dict of param_name-> float
            extracted from fit model
        hyper-parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_failed : bool
            The estimator failed to fit.
    """
    if not isinstance(error_score, numbers.Number) and error_score != 'raise':
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    X_sample = _safe_indexing(X, sample_indx)
    y_sample = _safe_indexing(y, sample_indx)

    result = {}
    try:
        if groups is True:
            estimator.fit(X_sample, y_sample, groups=sample_indx)
        else:
            estimator.fit(X_sample, y_sample)

    except Exception as e:
        if error_score == 'raise':
            raise e
        else:
            if isinstance(parameter_extractor, dict):
                parameters = {name: error_score for name in parameter_extractor}
            else:
                parameters = error_score
            result["fit_failed"] = True
            result["error"] = str(e)

    else:
        result["fit_failed"] = False
        parameters = _extract_parameters(parameter_extractor, estimator, error_score)

    result["parameters"] = parameters

    if return_estimator:
        result["estimator"] = estimator
    return result


def _extract_parameters(parameter_extractor, estimator, error_value):
    if isinstance(parameter_extractor, dict):
        try:
            result = {}
            for key, func in parameter_extractor.items():
                result[key] = func(estimator)

        except Exception:
            if error_value == 'raise':
                raise
            else:
                result = {k: error_value for k in parameter_extractor.keys()}
    else:
        try:
            result = parameter_extractor(estimator)
        except Exception:
            if error_value == 'raise':
                raise
            else:
                result = error_value

    return result
