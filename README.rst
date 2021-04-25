******
fftwpp
******

fftwpp is a header-only, C++ interface to the FFTW_ library. The goal of this
project is to expose an idiomatic API that remains as close as possible to the
underlying C library. The library is documented at
https://sbrisard.github.io/fftwpp/.

fftwpp is released under a BSD 3-Clause License.


Features
========

- Creation of plans follow the RAII paradigm
- Complex-to-complex, real-to-complex, complex-to-real transforms
- Optionally multi-threaded (OpenMP)
- Python bindings with pybind11_


News
====

- 25 apr. 2021 — Version 1.0 is released (including C-to-C, R-to-C, C-to-R and
  OpenMP).

.. _FFTW: http://fftw.org/
.. _pybind11: https://pybind11.readthedocs.io/
