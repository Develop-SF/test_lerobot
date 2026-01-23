# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------
project = 'LeRobot Testing Branch'
copyright = '2026, LeRobot Team'
author = 'LeRobot Team'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_suppress_warnings = ['myst.xref.missing']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_js_files = [
    'theme_init.js',
]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Logo and favicon
html_logo = None
html_favicon = None

# -- Source file settings ----------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Master document
master_doc = 'index'
