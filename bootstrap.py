
import numpy as np
from sklearn.utils import indexable, _safe_indexing
from joblib import Parallel
from sklearn.utils.fixes import delayed
from sklearn.base import clone
import numbers


def bootstrap_samples(n, r):
    """Return a generator sampling from range(n) r times."""
    indices = np.arange(n)
    for i in range(r):
        yield np.random.choice(indices, size=n, replace=True)


def bootstrap(estimator, X, y=None, parameter_extractor=None, samples=100,
              n_jobs=None, verbose=0,
              pre_dispatch='2*n_jobs',
              return_estimator=False, error_score=np.nan, groups = True):
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
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.

    Examples
    --------


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

    return results


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
