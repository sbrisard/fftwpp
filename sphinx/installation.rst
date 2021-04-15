************
Installation
************


Installing the C++ library
==========================

``fftwpp`` depends on FFTW_.

This is a CMake_ based project. The installation procedure is
standard. However, FFTW_ is not detected automatically by CMake_ and
you might need to specify its location manually (follow the CMake_
instructions if needed).

First, clone the repository. Then, ``cd`` into the root directory of
the ``fftwpp`` project. Let ``fftwpp_INSTALL_PREFIX`` be the path to
the directory where ``fftwpp`` should be installed::

  $ git clone https://github.com/sbrisard/fftwpp
  $ cd fftwpp
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=fftwpp_INSTALL_PREFIX ..
  $ cmake --build . --config Release
  $ cmake --install . --config Release

.. note:: The ``--config`` option might not be available, depending on
   the selected generator.

At this point, ``fftwpp`` should be installed. To check your
installation, you could try to :ref:`compile a program
<20210415083504>`.

To run the tests or build the documentation properly, you need to
first build the python bindings (see :ref:`below
<20210415083658>`).

.. _20210415083658:

Installing the Python bindings
==============================

To install the ``pyfftwpp`` module, ``cd`` into the ``python``
subdirectory and edit or create the ``setup.cfg``. You need to specify
the location of

- the ``pyfftwpp`` headers, see ``include_dir`` in section ``[fftwpp]``,
- the FFTW_ headers, see ``include_dir`` in section ``[fftw]``,
- the FFTW_ binaries, see ``library_dir`` in section ``[fftw]``.

The resulting file should look like::

  [pyfftwpp]
  include_dir = ${CMAKE_INSTALL_PREFIX}/include
  [fftw]
  include_dir = path/to/FFTW/headers
  library_dir = path/to/FFTW/binaries

Then, issue the following command::

  $ python setup.py install --user

or (if you intend to edit the project)::

  $ python setup.py develop --user

To run the tests with Pytest_::

  $ python -m pytest tests


Building the documentation
==========================

The documentation of ``fftwpp`` requires Sphinx_. The C++ API docs are
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
.. _Pytest: https://docs.pytest.org/
.. _Sphinx: https://www.sphinx-doc.org/
