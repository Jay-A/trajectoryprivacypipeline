import os
import sys

project = "Trajectory Privacy Pipeline"
author = "Jay A"
release = "0.1"

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "myst_parser",
]

html_theme = "sphinx_rtd_theme"
