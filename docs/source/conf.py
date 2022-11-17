# -*- coding: utf-8 -*-
import os
import sys

import rassine

__version__ = rassine.__version__

# from pkg_resources import DistributionNotFound, get_distribution
#
# try:
#     __version__ = get_distribution("emcee").version
# except DistributionNotFound:
#     __version__ = "unknown version"


# General stuff
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx.ext.graphviz",
    "sphinxarg.ext",  # to document command line arguments
]

graphviz_output_format = "svg"


# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}
autoclass_content = "class"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
autodoc_typehints = "signature"
autodoc_preserve_defaults = True  # does not work fully, but why not
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures
autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ["_templates"]
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
simplify_optional_unions = False
napoleon_include_init_with_doc = True
napoleon_use_rtype = False
napoleon_use_admonition_for_examples = True
# napoleon_preprocess_types = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True


source_suffix = ".rst"
master_doc = "index"

# myst_nb
myst_enable_extensions = ["dollarmath", "colon_fence"]

# sphinxcontrib.bibtex
bibtex_bibfiles = ["refs.bib"]

project = "rassine"
copyright = "2019-2022, Michael Cretignier, Xavier Dumusque, Denis Rosset & contributors"  # TODO: are the credits correct?
version = __version__
release = __version__
exclude_patterns = ["_build"]

# HTML theme
html_theme = "furo"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "RASSINE"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
nb_execution_mode = "off"
nb_execution_timeout = -1
html_extra_path = ["robots.txt"]
