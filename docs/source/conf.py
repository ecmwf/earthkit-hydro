# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

on_rtd = os.environ.get("READTHEDOCS") == "True"

if on_rtd:
    version = os.environ.get("READTHEDOCS_VERSION", "latest")
    release = version
else:
    version = "dev"
    release = "dev"

rtd_version = version if version != "latest" else "develop"
rtd_version_type = os.environ.get("READTHEDOCS_VERSION_TYPE", "branch")

if rtd_version_type in ("branch", "tag"):
    source_branch = rtd_version
else:
    source_branch = "main"

sys.path.insert(0, os.path.abspath("../../src"))

project = "earthkit-hydro"
copyright = "2025, European Centre for Medium-Range Weather Forecasts (ECMWF)"
author = "European Centre for Medium-Range Weather Forecasts (ECMWF)"
# release = '0.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Automatically extracts documentation from your Python docstrings
    "sphinx.ext.autodoc",
    # Supports Google-style and NumPy-style docstrings
    "sphinx.ext.napoleon",
    # Renders LaTeX math in HTML using MathJax
    "sphinx.ext.mathjax",
    # Option to click viewcode
    "sphinx.ext.viewcode",
    # Links to the documentation of other projects via cross-references
    # "sphinx.ext.intersphinx",
    # Generates summary tables for modules/classes/functions
    # "sphinx.ext.autosummary",
    # Allows citing BibTeX bibliographic entries in reStructuredText
    "sphinxcontrib.bibtex",
    # Tests snippets in documentation by running embedded Python examples
    # "sphinx.ext.doctest",
    # Checks documentation coverage of the codebase
    # "sphinx.ext.coverage",
    # Adds .nojekyll file and helps configure docs for GitHub Pages hosting
    # "sphinx.ext.githubpages",
    # Adds "Edit on GitHub" links to documentation pages
    # "edit_on_github",
    # Adds "Edit on GitHub" links to documentation pages
    # "sphinx_github_style",
    # Option to link to code
    # "sphinx.ext.linkcode",
    # Automatically includes type hints from function signatures into the documentation
    # "sphinx_autodoc_typehints",
    # Integrates Jupyter Notebooks into Sphinx
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "furo"

html_static_path = ["_static"]

# html_context = {
#     "display_github": True,
#     "github_user": "ecmwf",         # GitHub username
#     "github_repo": "docsample",             # GitHub repository name
#     "github_version": "main",               # Branch (e.g., 'main', 'master')
#     "conf_py_path": "/docs/",               # Path to your docs root in the repo
# }

bibtex_bibfiles = ["references.bib"]

html_theme_options = {
    "light_css_variables": {
        "color-sidebar-background": "#001F3F",
        "color-sidebar-link-text": "#ffffff",
        "color-sidebar-brand-text": "#ffffff",
        "color-sidebar-caption-text": "#ffffff",
        "color-brand-primary": "#00D9FF",
        "color-brand-content": "#5f8dd3",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#001F3F",
        "color-sidebar-link-text": "#ffffff",
        "color-sidebar-brand-text": "#ffffff",
        "color-sidebar-caption-text": "#ffffff",
        "color-brand-primary": "#00D9FF",
        "color-brand-content": "#5f8dd3",
    },
    "light_logo": "earthkit-hydro-dark.svg",
    "dark_logo": "earthkit-hydro-dark.svg",
    "source_repository": "https://github.com/ecmwf/earthkit-hydro/",
    "source_branch": source_branch,
    "source_directory": "docs/source",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ecmwf/earthkit-hydro",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
