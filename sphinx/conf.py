import pathlib

project_root = pathlib.Path.cwd().parent


def get_metadata(key):
    with open(project_root / "metadata" / (key + ".txt"), "r", encoding="utf8") as f:
        return f.read()


try:
    import pyfftwpp

    project = pyfftwpp.metadata["name"]
    author = pyfftwpp.metadata["author"]
    copyright = pyfftwpp.metadata["year"] + ", " + pyfftwpp.metadata["author"]
    release = pyfftwpp.metadata["version"]
except ImportError:
    from sphinx.util import logging

    logger = logging.getLogger(__name__)
    logger.warn("Could not import python module; some metadata is missing")
    project = "**MISSING PROJECT NAME**"
    copyright = "**MISSING COPYRIGHT**"
    author = "**MISSING AUTHOR**"
    release = "**MISSING VERSION**"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.todo", "breathe"]

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

todo_include_todos = True


breathe_projects_source = {
    "fftwpp": (
        str(project_root / "include" / "fftwpp"),
        ["fftwpp.hpp"],
    )
}

breathe_doxygen_config_options = {"GENERATE_TODOLIST": "YES"}

html_theme = "sphinxdoc"
html_static_path = ["_static"]
