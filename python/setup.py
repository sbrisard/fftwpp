import configparser
import os.path
import pathlib
import re

import pybind11

import setuptools
from setuptools.command.build_ext import build_ext


def read_metadata():
    metadata = {}
    filename = pathlib.Path.cwd() / ".." / "include" / "fftwpp" / "fftwpp.hpp"
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
    prog = re.compile(
        r"constexpr\s*std::string_view\s*([a-z]*)\s*\{\s*\"([^\"]*)\"\s*}\s*;"
    )
    for line in lines:
        result = prog.match(line)
        if result is not None:
            metadata[result.group(1)] = result.group(2)
    return metadata


class my_build_ext(build_ext):

    def build_extensions(self):
        fftw_libraries = {"msvc": ["fftw3"], "unix": ["fftw3"]}
        extra_compile_args = {
            "msvc": ["/std:c++17"],
            "unix": ["-std=c++17"],
        }
        if self.with_openmp:
            fftw_libraries["unix"].append("fftw3_omp")
            extra_compile_args["unix"].append("-fopenmp")
            extra_compile_args["msvc"].append("/openmp")

        for extension in self.extensions:
            extension.libraries += fftw_libraries[self.compiler.compiler_type]
            extension.extra_compile_args += extra_compile_args[
                self.compiler.compiler_type
            ]
        build_ext.build_extensions(self)


if __name__ == "__main__":
    metadata = read_metadata()

    with open(os.path.join("..", "README.md"), "r") as f:
        metadata["long_description"] = f.read()

    print(metadata)

    config = configparser.ConfigParser()
    config.read("setup.cfg")

    include_dirs = [pybind11.get_include()]
    library_dirs = []
    if config.has_section("fftwpp"):
        include_dirs.append(config["fftwpp"].get("include_dir", ""))
    if config.has_section("fftw"):
        include_dirs.append(config["fftw"].get("include_dir", ""))
        library_dirs.append(config["fftw"].get("library_dir", ""))

    # TODO Defining a class variable dynamically is poor practice
    my_build_ext.with_openmp = config["fftw"].getboolean("with_openmp")

    pyfftwpp = setuptools.Extension(
        "pyfftwpp",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        sources=["pyfftwpp.cpp"],
        language="c++",
    )

    setuptools.setup(
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        ext_modules=[pyfftwpp],
        cmdclass={"build_ext": my_build_ext},
        **metadata
    )
