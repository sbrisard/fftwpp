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


Example
=======

Here is a shortened, C++ example.

.. code:: cpp

   #include <fftwpp/fftwpp.hpp>

   // Create input and output arrays
   auto factory = fftwpp::PlanFactory().set_estimate();
   auto plan = factory.create_plan(rank, shape, in.data(), out.data(), -1);
   // Fill the input array
   plan.execute()

And the Python equivalent

.. code:: python

   import pyfftwpp

   # Create input and output arrays
   factory = pyfftwpp.PlanFactory().set_estimate()
   plan = factory.create_plan(rank, input, output, -1)
   # Fill the input array
   plan.execute()


News
====

- 25 apr. 2021 — Version 1.0 is released (including C-to-C, R-to-C, C-to-R and
  OpenMP).


Roadmap
=======

- Add support for MPI
- Provide idiomatic support for ``fftw_malloc/fftw_free`` (or any aligned memory
  allocation)


.. _FFTW: http://fftw.org/
.. _pybind11: https://pybind11.readthedocs.io/
