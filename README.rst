******
fftwpp
******

fftwpp is a header-only, C++ interface to the FFTW_ library. The goal of this
project is to expose an idiomatic API that remains as close as possible to the
underlying C library. The library is documented at
https://sbrisard.github.io/fftwpp/.

Features
========

- Creation of plans follow the RAII paradigm
- Complex-to-complex, real-to-complex, complex-to-real transforms
- Optionally multi-threaded (OpenMP)
- Python bindings with pybind11_

.. _FFTW: http://fftw.org/
.. _pybind11: https://pybind11.readthedocs.io/
