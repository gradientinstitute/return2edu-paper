# Impact of returning to education
Research project estimating the causal impact of further education on employment and wellbeing outcomes. 


## Install required python packages
   - install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
   - `conda env create -f environment.yml`

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
   - Optionally put any exposition markdown that should be displayed in a header page, rather than within every result page in a cell tagged `overview`.
   - Add the notebook and corresponding list of parameter settings to `execute_notebooks.py`

### Running parameterised notebooks and generating html
The code for this is all contained in `execute_notebooks`. The rough process is;

   1. Start with a Dict from Notebook to a list of parameter settings for that notebook.
   2. For each base notebook, parameter setting pair, Papermill generates and executes a notebook with those parameter settings.
   3. Custom code modifies some markdown components of the generated notebook to make them display better in Jupyterbook, generates a `_toc.yml` file and markdown files for chapter headings to group a set of parameterised results. 
   4. `jupyter-book` translates the generated notebooks into html. 

To run models and generate html, run `python execute_notebooks.py`
