************
Installation
************


Installing the C++ library
==========================

fftwpp depends on FFTW_.

This is a CMake_ based project. The installation procedure is
standard. However, FFTW_ is not detected automatically by CMake_ and
you might need to specify its location manually (follow the CMake_
instructions if needed).

First, clone the repository. Then, ``cd`` into the root directory of
the fftwpp project. Let ``fftwpp_INSTALL_PREFIX`` be the path to the
directory where fftwpp should be installed::

  $ git clone https://github.com/sbrisard/fftwpp
  $ cd fftwpp
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=fftwpp_INSTALL_PREFIX ..
  $ cmake --build . --config Release
  $ cmake --install . --config Release

.. note:: The ``--config`` option might not be available, depending on
   the selected generator.

At this point, fftwpp should be installed. To check your installation,
you could try to :ref:`compile a program <20210415083504>`.

To run the tests or build the documentation properly, you need to
first build the python bindings (see :ref:`below
<20210415083658>`).

.. _20210415083658:

Installing the Python bindings
==============================

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
.. _Pytest: https://docs.pytest.org/
.. _Sphinx: https://www.sphinx-doc.org/
