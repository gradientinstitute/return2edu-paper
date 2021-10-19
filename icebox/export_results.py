
from traitlets.config import Config
from nbconvert.exporters import HTMLExporter

import re
from traitlets import List, Unicode
from nbconvert.preprocessors import Preprocessor


class ResultPreprocessor(Preprocessor):
    """Identical to RegexRemoverPreprocessor except that matching cells are kept"""

    keep_both = Unicode(default_value='##![ALL]').tag(config=True)
    keep_output = Unicode(default_value='##![RESULT]').tag(config=True)

    def process_cell(self, cell):
        """
        Make any required modifications to the cell. 

        Returns: Boolean.
            False means the entire cell should be dropped from the output.
        """
        if cell['cell_type'] == 'markdown':  # markdown cells are included unless tagged hide
            return True

        # cell['outputs']
        # Compile all the patterns into one: each pattern is first wrapped
        # by a non-capturing group to ensure the correct order of precedence
        # and the patterns are joined with a logical or

        if cell.source.startswith(self.keep_both):
            return True

        if cell.source.startswith(self.keep_output):
            cell.source = ''
            return True

        return False

    def preprocess(self, nb, resources):
        """
        Preprocessing to apply to each notebook.
        """

        # Filter out cells that meet the conditions
        cells_to_keep = []
        for cell in nb.cells:
            print(cell['cell_type'])
            if self.process_cell(cell):
                cells_to_keep.append(cell)
        print("Keeping: ", len(cells_to_keep))
        nb.cells = cells_to_keep

        #nb.cells = [cell for cell in nb.cells if self.check_conditions(cell)]

        return nb, resources

# def run_notebook(notebook_path):
#     """Run a notebook using nbconvert and returns any errors that occur as a list."""
#     with open(notebook_path) as f:
#         nb = nbformat.read(f, as_version=4)

#     proc = ExecutePreprocessor(timeout=6000, kernel_name="python3")
#     proc.allow_errors = True
#     proc.preprocess(nb, {"metadata": {"path": "solutions/"}})

#     errors = []
#     for cell in nb.cells:
#         if "outputs" in cell:
#             for output in cell["outputs"]:
#                 if output.output_type == "error":
#                     errors.append(output)

# c =  Config()
# c.PelicanSubCell.start = 4
# c.PelicanSubCell.end = 6
# c.RSTExporter.preprocessors = [PelicanSubCell]

# # Create our new, customized exporter that uses our custom preprocessor
# pelican = RSTExporter(config=c)


# Process the notebook
# print(pelican.from_notebook_node(jake_notebook)[0])
if __name__ == "__main__":
    # Setup config
    c = Config()

    # Configure and run out exporter
    c.HTMLExporter.preprocessors = [ResultPreprocessor]

    exporter = HTMLExporter(config=c)
    exporter.register_preprocessor(ResultPreprocessor(config=c), True)

    # Configure and run our exporter - returns a tuple - first element with html,
    # second with notebook metadata
    output = HTMLExporter(config=c).from_filename("Data-Processing.ipynb")

    # Write to output html file
    with open("Data-Processing-results.html",  "w") as f:
        f.write(output[0])
