"""
Finn's run configuration for execute_notebooks.

A valid config must expose `NOTEBOOKS`, `RESULT_DIRECTORY` and `FORCE_EXECUTION`
"""

from reed import RunConfig, Notebook


RESULT_DIRECTORY = 'results'
FORCE_EXECUTION = False  # If False notebooks that already exist in results will not be re-executed


test = True
outcome = 'y_wsce'
treatment = 'redufl'
# A map from a notebook to a list of configurations to run that notebook with {Notebook -> List[RunConfig]}
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
    Notebook("Direct-Regression.ipynb", "Direct Regression"): [
        RunConfig("all-vars", {
            'configuration_name': "all-vars",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': "data/all_vars.csv"
        }),

        RunConfig("lasso-100", {
            'configuration_name': "lasso-100",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': "data/all_lasso_selected_100.csv"
        }),
        RunConfig("lasso-50", {
            'configuration_name': "lasso-50",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': "data/all_lasso_selected_50.csv"
        }),

        RunConfig("lasso-20", {
            'configuration_name': "lasso-20",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': "data/all_lasso_selected_20.csv"
        }),

    ],
    Notebook("EconML2.ipynb", "EconML"): [
        RunConfig("econml", {
            'configuration_name': "econ-lasso-20",
            'outcome': outcome,
            'treatment': treatment,
            'test': test,
            'data_file': "data/all_lasso_selected_20.csv"
        })
    ]
}
