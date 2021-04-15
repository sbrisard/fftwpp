import configparser
import os.path

import pybind11

import setuptools
from setuptools.command.build_ext import build_ext


def get_metadata(key):
    with open(os.path.join("..", "metadata", key + ".txt"), "r", encoding="utf8") as f:
        return f.read().strip()


class build_ext_cpp17(build_ext):
    args = {"msvc": "/std:c++17", "unix": "-std=c++17"}

    def build_extensions(self):
        print(f"***{self.compiler.compiler_type}***")
        for extension in self.extensions:
            extension.extra_compile_args.append(self.args[self.compiler.compiler_type])
        build_ext.build_extensions(self)


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

    include_dirs = [pybind11.get_include()]
    library_dirs = []
    if config.has_section("fftwpp"):
        include_dirs.append(config["fftwpp"].get("include_dir", ""))
    if config.has_section("fftw"):
        include_dirs.append(config["fftw"].get("include_dir", ""))
        library_dirs.append(config["fftw"].get("library_dir", ""))

    pyfftwpp = setuptools.Extension(
        "pyfftwpp",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["fftw3"],
        sources=["pyfftwpp.cpp"],
        # define_macros=[
        #     ("__FFTWPP_VERSION__", r"\"" + metadata["version"] + r"\""),
        #     ("__FFTWPP_AUTHOR__", r"\"" + metadata["author"] + r"\""),
        # ],
        language="c++"
    )

    setuptools.setup(
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        ext_modules=[pyfftwpp],
        cmdclass={"build_ext": build_ext_cpp17},
        **metadata
    )
