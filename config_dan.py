"""
Dan's run configuration for execute_notebooks.

A valid config must expose `NOTEBOOKS`, `RESULT_DIRECTORY` and `FORCE_EXECUTION`
"""

from reed import RunConfig, Notebook


RESULT_DIRECTORY = 'results'
FORCE_EXECUTION = False  # If False notebooks that already exist in results will not be re-executed

test = False
outcome = 'y_wsce'
treatment = 'redufl'
release = 'restricted'
laplace_prior = False
xval_gp = True

# A map from a notebook to a list of configurations to run that notebook with
# {Notebook -> List[RunConfig]}
NOTEBOOKS = {

    Notebook("Data-Processing.ipynb", "Data Processing"): [
        RunConfig('general', {'test': test, 'release': 'general'})
    ],
    Notebook("Compare-Anna-Treatment-Outcomes.ipynb", "Check Treatment and Outcome Coding"): [
        RunConfig('default', {})
    ],
    Notebook("OLS-basic-vars.ipynb", "Basic OLS"): [
        RunConfig('default', {'test': test, 'outcome': outcome, 'treatment': treatment})
    ],
    Notebook("Direct-Bayesian.ipynb", "Direct Bayesian"): [
        RunConfig(f"all-vars-{release}", {
            'configuration_name': f"all-vars-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_vars_{release}.csv",
            'release': release,
            'laplace_prior': laplace_prior,
            'xval_gp': xval_gp
        }),

        RunConfig(f"lasso-100-{release}", {
            'configuration_name': f"lasso-100-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_lasso_selected_100_{release}.csv",
            'release': release,
            'laplace_prior': laplace_prior,
            'xval_gp': xval_gp
        }),
        RunConfig(f"lasso-50-{release}", {
            'configuration_name': f"lasso-50-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_lasso_selected_50_{release}.csv",
            'release': release,
            'laplace_prior': laplace_prior,
            'xval_gp': xval_gp
        }),

        RunConfig(f"lasso-20-{release}", {
            'configuration_name': f"lasso-20-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_lasso_selected_20_{release}.csv",
            'release': release,
            'laplace_prior': laplace_prior,
            'xval_gp': xval_gp
        }),

    ],
    Notebook("Direct-Regression.ipynb", "Direct Regression"): [
        RunConfig(f"all-vars-{release}", {
            'configuration_name': f"all-vars-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_vars_{release}.csv",
            'release': release
        }),

        RunConfig(f"lasso-100-{release}", {
            'configuration_name': f"lasso-100-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_lasso_selected_100_{release}.csv",
            'release': release
        }),
        RunConfig(f"lasso-50-{release}", {
            'configuration_name': f"lasso-50-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_lasso_selected_50_{release}.csv",
            'release': release
        }),

        RunConfig(f"lasso-20-{release}", {
            'configuration_name': f"lasso-20-{release}",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_lasso_selected_20_{release}.csv",
            'release': release
        }),

    ],
    Notebook("EconML2.ipynb", "EconML"): [
        RunConfig(f"econml-{release}", {
            'configuration_name': "econ-lasso-20",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': f"data/all_lasso_selected_20_{release}.csv"
        })
    ]
}
