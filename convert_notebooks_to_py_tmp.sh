find . -name results -prune -o -name .ipynb_checkpoints -prune -o -name econml_notebooks -prune -o -name '*.ipynb' -exec jupyter nbconvert --to script --output-dir tmp {} \;
