import configparser
import os.path

import pybind11
import setuptools


def get_metadata(key):
    with open(os.path.join("..", "metadata", key+".txt"), "r", encoding="utf8") as f:
        return f.read().strip()


if __name__ == "__main__":
    metadata = {
        "name": "pyfftwpp",
        "version": get_metadata("version"),
        "author": get_metadata("author"),
        "author_email": "email",
        "description": get_metadata("description"),
        "url": get_metadata("repository"),
    }

    with open(os.path.join("..", "README.md"), "r") as f:
        metadata["long_description"] = f.read()

    config = configparser.ConfigParser()
    config.read("setup.cfg")
    fftwpp_include_dir = config["fftwpp"].get("include_dir", "")
    pyfftwpp = setuptools.Extension(
        "pyfftwpp",
        include_dirs=[pybind11.get_include(),
                      config["fftw"].get("include_dir", ""),
                      fftwpp_include_dir],
        library_dirs = [config["fftw"].get("library_dir", "")],
        libraries = ["fftw3"],
        sources=["pyfftwpp.cpp"],
        define_macros=[
            ("__FFTWPP_VERSION__", r"\"" + metadata["version"] + r"\""),
            ("__FFTWPP_AUTHOR__", r"\"" + metadata["author"] + r"\""),
        ],
        extra_compile_args=["/std:c++17"]
    )

    setuptools.setup(
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        ext_modules=[pyfftwpp],
        **metadata
    )
