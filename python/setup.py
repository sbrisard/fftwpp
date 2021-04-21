import configparser
import os.path
import pathlib
import re

import pybind11

import setuptools
from setuptools.command.build_ext import build_ext


def read_metadata():
    metadata = {}
    filename = pathlib.Path.cwd()/ ".." / "include"/ "fftwpp"/"fftwpp.hpp"
    with open(filename , "r", encoding="utf8") as f:
        lines = f.readlines()
    prog = re.compile(r"constexpr\s*std::string_view\s*([a-z]*)\s*\{\s*\"([^\"]*)\"\s*}\s*;")
    for line in lines:
        result = prog.match(line)
        if result is not None:
            metadata[result.group(1)] = result.group(2)
    return metadata


class build_ext_cpp17(build_ext):
    fftw_libraries = {"msvc": ["fftw3"], "unix": ["fftw3", "fftw3_omp"]}
    extra_compile_args = {"msvc": ["/std:c++17", "/openmp"], "unix": ["-std=c++17", "-fopenmp"]}

    def build_extensions(self):
        for extension in self.extensions:
            extension.libraries += self.fftw_libraries[self.compiler.compiler_type]
            extension.extra_compile_args += self.extra_compile_args[self.compiler.compiler_type]
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

    pyfftwpp = setuptools.Extension(
        "pyfftwpp",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
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
