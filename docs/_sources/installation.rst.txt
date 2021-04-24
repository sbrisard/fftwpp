************
Installation
************

First of all, clone the repository

.. code-block:: none

  $ git clone https://github.com/sbrisard/fftwpp


Installing the C++ library
==========================

fftwpp is a header-only library: there is no installation procedure *per se* and
you can drop the header wherever you like (as long as it is located in a
``fftwpp`` subdirectory). To use fftwpp in a C++ project, you must include the
header

.. code-block:: cpp

   #include <fftwpp/fftwpp.hpp>

and inform the compiler of its location.

.. note:: fftwpp depends on FFTW_. You must pass the relevant options to the
          compiler. Typically, these would be ``-I`` and ``-L`` options, as well
          as ``-lfftw3``. The C++ tutorials provides a
	  :ref:`CMake example <sec20210415083504>`.

.. note:: Optional compilation with OpenMP is automatically detected by
          fftwpp. If you *do* compile with the ``-fopenmp`` option (or
          equivalent), then you *must* link against the OpenMP FFTW
          library. This is typically done by passing the ``-lfftw3_omp`` to the
          linker. This automatically exposes the functions
          :cpp:func:`fftwpp::init_threads` and
          :cpp:func:`fftwpp::plan_with_nthreads`.

To run the tests or build the documentation properly, you need to first build
the python bindings (see :ref:`below <sec20210415083658>`).

.. _sec20210415083658:

Installing the Python bindings
==============================

The Python bindings are built with pybind11_, which must be installed.

To install the pyfftwpp module, ``cd`` into the ``python`` subdirectory and run
the ``setup.py`` script as follows.

First, build the extension::

  $ python setup.py build_ext -lfftw3

or, if you want to use the OpenMP version of FFTW::

  $ python setup.py build_ext -lfftw3 --with-openmp

.. note:: The ``--libraries/-l`` option is necessary to inform the linker of the
          name of the FFTW library.

.. note:: You might need to inform the compiler of the location of the FFTW
          header (``--include-dirs/-I`` option) and binaries
          (``--library-dirs/L`` option).

On my computer (Windows 10 platform with Miniconda3), the build command reads::

  $ python setup.py build_ext -IC:\Users\sbrisard\miniconda3\Library\include -LC:\Users\sbrisard\miniconda3\Library\lib -lfftw3

When the extension is built, installation is down as usual::

  $ python setup.py install --user

or (if you intend to edit the project)::

  $ python setup.py develop --user

To run the tests with Pytest_::

  $ python -m pytest tests


Building the documentation
==========================

.. note:: For the documentation to build properly, the python module
          must be installed, as it is imported to retrieve the project
          metadata.

The documentation of fftwpp requires Sphinx_. The C++ API docs are
built with Doxygen_ and the Breathe_ extension to Sphinx_.

To build the HTML version of the docs in the ``public`` subdirectory::

  $ cd docs
  $ sphinx-build -b html . ../public

To build the LaTeX version of the docs::

  $ cd docs
  $ make latex


.. _Breathe: https://breathe.readthedocs.io/
.. _CMake: https://cmake.org/
.. _Doxygen: https://www.doxygen.nl/
.. _FFTW: http://fftw.org/
.. _pybind11: https://pybind11.readthedocs.io/
.. _Pytest: https://docs.pytest.org/
.. _Sphinx: https://www.sphinx-doc.org/
