"""
Execute parameterised notebooks
"""
import papermill as pm
import nbformat
from dataclasses import dataclass
import os
import os.path
import subprocess
from collections import defaultdict


@dataclass(frozen=True)
class RunConfig:
    """Parameter settings for notebook."""
    name: str
    parameters: dict


@dataclass(frozen=True)
class Notebook:
    filename: str
    title: str


def drop_ext(name, ext):
    if name.endswith(ext):
        return name[:-len(ext)]
    return name


def safeget(dct, *keys):
    """Return value in nested dict or None if any part of the path is not there."""
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct


def generate_toc_yml(notebooks: {Notebook: [RunConfig]}, directory: str) -> None:
    """
    Generate the _toc.yml file and any required markdown files to mark chapters.
    """
    filepath = os.path.join(directory, "_toc.yml")
    contents = "format: jb-book\nroot: intro\nchapters:\n"
    for notebook, configurations in notebooks.items():
        notebook_name = drop_ext(notebook.filename, '.ipynb')

        if len(configurations) > 1:
            # Make a markdown file to create a chapter for this notebook
            chapter_filepath = os.path.join(directory, f'{notebook_name}.md')
            with open(chapter_filepath, 'w') as chapter_intro:
                chapter_intro.write(f'# {notebook.title}\n')

            # Make an entry for the chapter in _toc.yml
            contents += f"- file: {notebook_name}\n"

            # Make each parameterised variant of notebook a section
            contents += "  sections:\n"
            for config in configurations:
                _, output_name = _output_notebook_path(notebook, config, directory)
                contents += f"  - file: {output_name}\n"

        else:  # only one configuration - doesn't need subheadings
            config = configurations[0]
            _, output_name = _output_notebook_path(notebook, config, directory)
            contents += f"- file: {output_name}\n"

    with open(filepath, 'w') as out:
        out.write(contents)


def _output_notebook_path(notebook: Notebook, config: RunConfig, directory: str) -> str:
    """Return the path to save the parameterised notebook to."""
    ext = '.ipynb'
    base_name = drop_ext(notebook.filename, ext)
    output_name = f'{base_name}-{config.name}'
    output_path = os.path.join(directory, f'{output_name}{ext}')
    return output_path, output_name


def run_notebooks(
    notebooks: {Notebook: [RunConfig]}, result_dir: str, skip_if_exists: bool = False
) -> {Notebook: [RunConfig]}:
    """
    Execute each configuration of the notebooks using Papermill.

    Parameters
    ----------
    notebooks:
        Map from the notebook to the list of configurations to execute for that notebook

    result_dir:
        The directory to save the generated notebooks to.

    """
    executed = defaultdict(list)
    for notebook, configurations in notebooks.items():
        for config in configurations:
            output_path, _ = _output_notebook_path(notebook, config, result_dir)
            if not os.path.exists(output_path) or not skip_if_exists:
                print("executing: ", notebook.filename, "saving to ", output_path)
                executed[notebook].append(config)
                pm.execute_notebook(
                    notebook.filename,
                    output_path,
                    parameters=config.parameters
                )
    return executed


def find_overview_indx(nb):
    """find the first cell marked overview"""
    for indx, cell in enumerate(nb.cells):
        tags = safeget(cell, 'metadata', 'tags')
        if tags is not None and 'overview' in tags:
            return indx
    return None


def set_result_notebook_title(notebooks, result_dir):
    """Add a title cell to each of the generated notebooks so they are linked by jupyter-book."""
    for notebook, configurations in notebooks.items():
        if len(configurations) == 1:
            continue
        content = None
        for config in configurations:
            output_path, _ = _output_notebook_path(notebook, config, result_dir)
            nb = nbformat.read(output_path, as_version=4)
            overview_indx = find_overview_indx(nb)
            if overview_indx is not None:
                content = nb.cells[overview_indx]['source']
                nb.cells.pop(overview_indx)
            nb.cells.insert(0, nbformat.v4.new_markdown_cell(f"# {config.name}"))

            with open(output_path, 'w') as f:
                nbformat.write(nb, f)

        # write overview content to the markdown file
        if content is not None:
            notebook_name = drop_ext(notebook.filename, '.ipynb')
            chapter_markdown = os.path.join(result_dir, f'{notebook_name}.md')
            with open(chapter_markdown, 'a') as f:
                f.write(content)


if __name__ == "__main__":

    outcome = 'y_wsce'
    treatment = 'redufl'
    RESULT_DIRECTORY = 'results'
    test = False
    force_execution = False  # If False notebooks that already exist in results will not be re-executed

    # A map from a notebook to a list of configurations to run that notebook with {Notebook -> List[RunConfig]}
    NOTEBOOKS = {

        Notebook("Data-Processing.ipynb", "Data Processing"): [RunConfig('default', {'test': test})],
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

    generated_notebooks = run_notebooks(
        NOTEBOOKS, RESULT_DIRECTORY, skip_if_exists=(not force_execution))
    generate_toc_yml(NOTEBOOKS, RESULT_DIRECTORY)
    set_result_notebook_title(generated_notebooks, RESULT_DIRECTORY)

    subprocess.run(["jupyter-book build results/"], shell=True)
