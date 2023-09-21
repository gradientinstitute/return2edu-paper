# Impact of returning to education
Research project estimating the causal impact of further education on employment and wellbeing outcomes. 

This code depends on the use having access to the [Household, Income and Labour
Dynamics in Australia (HILDA)](https://melbourneinstitute.unimelb.edu.au/hilda)
dataset. Please see the website for access conditions and procedures.


## Install required python packages
   - install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
   - `conda env create -f environment.yml`

## Place data in the data directory
   - `Data_preparation.py` requires the HILDA files `part1/Combined {...}190c.sav` to be placed in the data directory.

## View or run models
   - run `conda activate reed`
   - within the re-education directory run `jupyter notebook`

## Parameterising notebooks and building result HTML

There are many cases where we want to run models and display results over different datasets. 
Notebooks are good at keeping results, plots and the code used to generate them together, but 
we want to avoid having multiple near-identical notebooks for different inputs. To solve this issue, 
notebooks can be parameterised (using Papermill). 

### Parameterising notebooks
   - Put all the paramters in a single code cell. Tag the cell with `parameters`.
   - Optionally put any exposition markdown that should be displayed in a header page, rather than within every result page, in a cell tagged `overview`.
   - Add the notebook and corresponding list of parameter settings to `execute_notebooks.py`

### Running parameterised notebooks and generating html
The code for this is all contained in `execute_notebooks`. The rough process is;

   1. Write a configuration file containing a Dict from Notebook to a list of parameter settings for that notebook.
   2. For each base notebook, parameter setting pair, Papermill generates and executes a notebook with those parameter settings.
   3. Custom code modifies some markdown components of the generated notebook to make them display better in Jupyterbook, generates a `_toc.yml` file and markdown files for chapter headings to group a set of parameterised results. 
   4. `jupyter-book` translates the generated notebooks into html. 


#### Writing a run configuration file
The configuration file required by `execute_notebooks` is just a python file that exposes 3 global variables.
   - `RESULT_DIRECTORY` (the directory results should be saved in)
   - `FORCE_EXECUTION` (should existing results be overwritten)
   - `NOTEBOOKS` (a Dict from Notebook to a list of parameter settings for that notebook)

Below is a minimal example configuration that will execute the Data-Processing notebook in test mode using the general release. 

```
from reed import RunConfig, Notebook


RESULT_DIRECTORY = 'test_results'
FORCE_EXECUTION = False  # If False notebooks that already exist in results will not be re-executed

# A map from a notebook to a list of configurations to run that notebook with {Notebook -> List[RunConfig]}
NOTEBOOKS = {
    Notebook("Data-Processing.ipynb", "Data Processing"): [
        RunConfig('general', {'test': True, 'release': 'general'})
    ],
}
```
To run models and generate html, run `python execute_notebooks.py {your_config_here.py}`
