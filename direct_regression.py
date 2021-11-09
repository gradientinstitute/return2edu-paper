import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from reed import treatment_control_split, regex_select
from collections import defaultdict
import pickle
import time


def importance_from_coef(estimator):
    if hasattr(estimator, 'best_estimator_'):
        estimator = estimator.best_estimator_
    coef = estimator.coef_
    return {'importance': np.abs(coef), 'coef': coef}


def bootstrapped_cross_val(
    construct_models,
    cache_name,
    X0, X1, y0, y1,
    optimisation_metric,
    extract_params_func,
    inner_cv=3,
    load_from_cache=False,
    samples=10
):
    if load_from_cache:
        with open(cache_name, 'rb') as f:
            results = pickle.load(f)
    else:
        models0, models1 = construct_models(), construct_models()
        results = {}
        start = time.time()
        for model0, model1 in zip(models0, models1):
            print(f"Fitting {model0.name} ...", end='')
            results0 = model0.bootstrap_cv_evaluate(X0, y0,
                                                    optimisation_metric, extract_params_func,
                                                    inner_cv=inner_cv,
                                                    bootstrap_samples=samples,
                                                    return_estimator=True
                                                    )
            results1 = model1.bootstrap_cv_evaluate(X1, y1,
                                                    optimisation_metric, extract_params_func,
                                                    inner_cv=inner_cv,
                                                    bootstrap_samples=samples,
                                                    return_estimator=True
                                                    )

            results[model0.name] = (results0, results1)
            print("Done")
        total = time.time()-start
        print(f"Total time:{total} seconds")
        print(f"Caching results to: {cache_name}")
        with open(cache_name, 'wb') as f:
            pickle.dump(results, f)

    return results


def nested_cross_val(
    construct_models,
    cache_name,
    X0, X1, y0, y1,
    optimisation_metric,
    evaluation_metrics,
    innercv=None,
    outercv=None,
    load_from_cache=False,
):
    """

    Returns
    ----------
    models0: list
        The list of models used to fit the control surface
    models1: list
        The corresponding list of models used to fit the treatment surface
    results: {str: ({str:iterable},{str:iterable})}
        A dict from the model name to a tuple of results, the first for the control surface, the second for the treatment surface. 
    """
    if load_from_cache is True:
        if cache_name is None:
            raise ValueError("cache_name cannot be None when load_from_cache is True.")
        with open(cache_name, 'rb') as f:
            models0, models1, results = pickle.load(f)

    else:
        models0, models1 = construct_models(), construct_models()
        results = {}
        for model0, model1 in zip(models0, models1):
            print(f"Fitting {model0.name} ...", end='')
            results0 = model0.nested_cv_fit_evaluate(
                X0, y0, optimisation_metric, evaluation_metrics,
                inner_cv=innercv, outer_cv=outercv)
            results1 = model1.nested_cv_fit_evaluate(
                X1, y1, optimisation_metric, evaluation_metrics,
                inner_cv=innercv, outer_cv=outercv)
            results[model0.name] = (results0, results1)
            print("Done")

        if cache_name is not None:
            print(f"Caching results to {cache_name}")
            with open(cache_name, 'wb') as f:
                pickle.dump((models0, models1, results), f)

    return models0, models1, results


def exclude_vars():
    """Return a list of variables that should not be included as features."""
    treatments = ['^reduhl$', '^rehllt$', '^redudl$', '^redufl$', '^redllt$', '^refllt$']
    outcomes = ['^rlwage$', '^mh$', '^mhbm$', '^wkhr$', '^y_']
    other = [
        '^p_rcom',
        '^p_rdf',
        '^p_cotrl',
        '^xwaveid$',
        'p_rcom18'  # ?
        '^aedcq',  # indicate studying at start - these people should already have been removed
        '^abnfsty',
        '^aedcqfpt',
        '^aedqstdy'
    ]
    exclude = treatments + outcomes + other
    return exclude


def seperate_and_transform_data(data, treatment, outcome):

    # determine which variables are features
    exclude = regex_select(data.columns, exclude_vars()+[treatment, outcome])
    features = [c for c in data.columns if c not in exclude]

    transform = Pipeline([
        ('impute_missing', SimpleImputer()),
        # Making this RobustScaler totally kills performance on the full dataset - TODO understand why
        ('scale', StandardScaler()),
    ])

    # extract X
    X = data[features]
    n, m = X.shape
    X = transform.fit_transform(X)
    assert X.shape == (n, m), f"Transform changed data dimensions: {(n,m)} -> {X.shape}"

    # extract t and y
    y = data[outcome].copy().values
    t = data[treatment].copy().values

    # split X and y by t
    cntr = data[treatment] == 0
    treat = data[treatment] == 1

    X0, y0 = X[cntr, :], y[cntr]
    X1, y1 = X[treat, :], y[treat]

    return X0, X1, y0, y1, X, y, t, features


def print_unconditional_effects(data, treatment, y0, y1):
    print(f"Proportion Treated:{100*data[treatment].mean():.0f}%")
    print(f"Average outcome under Control:{y0.mean():.2f}±{y0.std()/np.sqrt(len(y0)):.2f}")
    print(f"Average outcome under Treatment:{y1.mean():.2f}±{y1.std()/np.sqrt(len(y1)):.2f}")
    print(f"Unadjusted treatment estimate {y1.mean() - y0.mean():.2f}")


def display_feature_importance(models0, models1, results, features, show=20):
    """

    """
    importances = {}
    for (m0, m1) in zip(models0, models1):
        if (m0.importance_func is not None) and (m1.importance_func is not None):
            cntr_result, treat_result = results[m0.name]
            i0 = m0.feature_importance(cntr_result, features, agg=False)
            i1 = m1.feature_importance(treat_result, features, agg=False)

            i = pd.merge(i0, i1, on=('fold_id', 'feature'), suffixes=('_cntr', '_treat'))
            i['importance'] = i[['importance_treat', 'importance_cntr']].mean(axis=1)
            agg_columns = ['importance']
            if 'coef_treat' in i.columns and 'coef_cntr' in i.columns:
                i['Δcoef'] = i['coef_treat']-i['coef_cntr']
                agg_columns.extend(['Δcoef', 'coef_treat', 'coef_cntr'])

            i = i.groupby('feature')[agg_columns].agg(('mean', 'std'))
            i.columns = ["_".join(col_name).rstrip('_') for col_name in i.columns.to_flat_index()]
            i.sort_values(by='importance_mean', ascending=False, inplace=True)
            print(m0.name)
            display(i.head(show))
            importances[m0.name] = i
    return importances


def estimate_average_causal_effect(X, model0, model1):
    """
    Return the average and standard deviation of estimated individual causal effects.

    Parameters
    ----------
    X: 2d array or pd.DataFrame
        the features/covariates on which to predict causal effects
    model0: estimator supporting .predict
        The model for predicting control values
    model1: estimator supporting .predict
        The model for predicting treated values
    """

    y0 = model0.predict(X)
    y1 = model1.predict(X)
    tau = y1-y0

    ate = tau.mean()
    sd_ate = tau.std()
    return ate, sd_ate


def estimate_causal_effect(X, models0, models1):
    """
    Return an array of ate estimates of length len(models0)=len(models1)

    Parameters
    ----------
    X: 2d array or pd.DataFrame
        the features/covariates on which to predict causal effects
    model0: iterable of estimators supporting .predict
        The models for predicting control values
    model1: iterable of estimators supporting .predict
        The models for predicting treated values
    """
    tau = []
    for e0, e1 in zip(models0, models1):
        y0 = e0.predict(X)
        y1 = e1.predict(X)
        tau.append(y1-y0)

    # array of shape len(modelsi),len(X)
    cate = np.array(tau)

    # array of shape len(modelsi) with the ate estimate for each sample
    ate = np.mean(cate, axis=1)
    return ate


def compute_ate(results, X, evaluation_metrics=None):
    if evaluation_metrics is None:
        evaluation_metrics = []
    rows = []
    index = []
    tau_estimates = {}
    for model_name, (contr_result, treat_result) in results.items():
        tau = estimate_causal_effect(X, contr_result['estimator'], treat_result['estimator'])
        tau_estimates[model_name] = tau
        row = {'ACE': tau.mean(), 'ACE_std': tau.std()}

        for m in evaluation_metrics:
            key = f'test_{m}'
            for name, result in [('control', contr_result), ('treated', treat_result)]:
                label = f"{name}_{m}"
                label_std = f"{label}_std"
                row[label] = np.mean(result[key])
                row[label_std] = np.std(result[key])
        rows.append(row)
        index.append(model_name)
    metrics = pd.DataFrame(rows, index=index)
    return metrics, tau_estimates


def visualise_ate(results, X, evaluation_metrics=None):
    """

    """
    metrics, tau_estimates = compute_ate(results, X, evaluation_metrics)

    with pd.option_context('display.float_format', '{:,.2f}'.format):
        display(metrics)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    ax[0].bar(metrics.index, metrics['control_r2'], yerr=metrics['control_r2_std'],
              align='center', alpha=0.5, capsize=10)
    ax[1].bar(metrics.index, metrics['treated_r2'], yerr=metrics['treated_r2_std'],
              align='center', alpha=0.5, capsize=10)
    ax[0].set_ylabel('$R^2$')
    ax[0].set_title('control model')
    ax[1].set_title('treated model')
    return metrics, tau_estimates


def plot_ate_distribution(tau_estimates):
    l = len(tau_estimates)
    fig, ax = plt.subplots(1, l, figsize=(5*l, 5))
    for i, (model, estimates) in enumerate(tau_estimates.items()):
        ax[i].hist(estimates)
        ax[i].set_xlabel('average causal effect')
        ax[i].set_ylabel('count')
        ax[i].set_title(model)
        ax[i].axvline(estimates.mean(), color='red')


def hyperparam_distributions(samples) -> {str: []}:
    """Returns a dict from hyper-parameter name to the best values for that hyper-parameter over the samples."""
    assert isinstance(
        samples, dict), 'samples should be a dict from model name to a list of results'

    distributions = defaultdict(list)
    bounds = defaultdict(lambda: [np.inf, -np.inf])
    estimators = samples['estimator']
    for e in estimators:
        h = e.best_params_
        grid = e.param_grid
        for key, value in h.items():
            distributions[key].append(value)
            if key in grid:
                search_space = grid[key]
                minv, maxv = np.min(search_space), np.max(search_space)
                if bounds[key][0] > minv:
                    bounds[key][0] = minv
                if bounds[key][1] < maxv:
                    bounds[key][1] = maxv
    return distributions, bounds


def plot_hyperparam_distributions(samples, title, show_bounds=True) -> None:
    distributions, bounds = hyperparam_distributions(samples)
    k = len(distributions)
    fig, axes = plt.subplots(1, k, figsize=(k*5, 4))
    if k == 1:
        axes = [axes]
    for i, (key, values) in enumerate(distributions.items()):
        ax = axes[i]
        ax.hist(values)
        ax.set_title(title)
        ax.set_xlabel(key)
        ax.set_ylabel('count')
        if show_bounds:
            xmin, xmax = ax.get_xlim()
            bound_min, bound_max = bounds[key]
            if bound_min > xmin:
                ax.axvline(bound_min, color="red")
            if bound_max < xmax:
                ax.axvline(bound_max, color="orange")

    return fig, axes
