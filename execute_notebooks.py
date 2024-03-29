"""
Execute parameterised notebooks
"""
import papermill as pm
import nbformat
import os
import os.path
import subprocess
import importlib
from collections import defaultdict
from reed import RunConfig, Notebook
import sys


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
    config_file = sys.argv[1]
    assert config_file.endswith(".py"), "config must be a .py file"
    config_module = config_file[0:-len('.py')]

    config = importlib.import_module(config_module)
    NOTEBOOKS = config.NOTEBOOKS
    RESULT_DIRECTORY = config.RESULT_DIRECTORY
    FORCE_EXECUTION = config.FORCE_EXECUTION

    generated_notebooks = run_notebooks(
        NOTEBOOKS, RESULT_DIRECTORY, skip_if_exists=(not FORCE_EXECUTION))
    generate_toc_yml(NOTEBOOKS, RESULT_DIRECTORY)
    set_result_notebook_title(generated_notebooks, RESULT_DIRECTORY)

    subprocess.run([f"jupyter-book build {RESULT_DIRECTORY}/"], shell=True)
