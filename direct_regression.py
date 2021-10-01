import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from reed import treatment_control_split, split_and_transform, regex_select
import pickle
import time


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
        models = construct_models()
        results = {},
        start = time.time()
        for model in models:
            print(f"Fitting {model.name} ...", end='')
            results0 = model.bootstrap_cv_evaluate(X0, y0,
                                                   optimisation_metric, extract_params_func,
                                                   inner_cv=inner_cv,
                                                   bootstrap_samples=samples,
                                                   return_estimator=True
                                                   )
            results1 = model.bootstrap_cv_evaluate(X1, y1,
                                                   optimisation_metric, extract_params_func,
                                                   inner_cv=inner_cv,
                                                   bootstrap_samples=samples,
                                                   return_estimator=True
                                                   )
            results[model.name] = (results0, results1)
            print("Done")
        total = time.time()-start
        print(f"Total time:{total} seconds")
        print(f"Caching results to: {cache_name}")
        with open(cache_name, 'wb') as f:
            pickle.dump(results, f)

    return results


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

    transform = Pipeline([
        ('impute_missing', SimpleImputer()),
        ('scale', StandardScaler()),
    ])

    exclude = exclude_vars()

    control, treated = treatment_control_split(data, treatment)
    features = regex_select(data.columns, exclude, exclude=True)
    X0, y0 = split_and_transform(control, features, outcome, transform)
    X1, y1 = split_and_transform(treated, features, outcome, transform)

    # construct the full dataset (remove ordering by treatment in case of any order dependance in fit)
    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))
    indx = np.arange(len(y))
    np.random.shuffle(indx)
    X = X[indx, :]
    y = y[indx]

    return X0, X1, y0, y1, X, y, features


def print_unconditional_effects(data, treatment, y0, y1):
    print(f"Proportion Treated:{100*data[treatment].mean():.0f}%")
    print(f"Average outcome under Control:{y0.mean():.2f}±{y0.std()/np.sqrt(len(y0)):.2f}")
    print(f"Average outcome under Treatment:{y1.mean():.2f}±{y1.std()/np.sqrt(len(y1)):.2f}")
    print(f"Unadjusted treatment estimate {y1.mean() - y0.mean():.2f}")


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
    if load_from_cache:
        with open(cache_name, 'rb') as f:
            models, results = pickle.load(f)

    else:
        models = construct_models()
        results = {}
        for model in models:
            print(f"Fitting {model.name} ...", end='')
            results0 = model.nested_cv_fit_evaluate(
                X0, y0, optimisation_metric, evaluation_metrics,
                inner_cv=innercv, outer_cv=outercv)
            results1 = model.nested_cv_fit_evaluate(
                X1, y1, optimisation_metric, evaluation_metrics,
                inner_cv=innercv, outer_cv=outercv)
            results[model.name] = (results0, results1)
            print("Done")

        print(f"Caching results to {cache_name}")
        with open(cache_name, 'wb') as f:
            pickle.dump((models, results), f)

    return models, results
