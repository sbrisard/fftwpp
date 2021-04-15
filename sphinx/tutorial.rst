********
Tutorial
********

This chapter provides a tutorial to the fftwpp project.

C++ tutorial
============

.. _20210415083504:

Compiling the program
---------------------

``cd`` into the ``example`` subdirectory. The provided example program
should be compiled and linked against ``fftwpp``::

  $ mkdir build
  $ cd build
  $ cmake -Dfftwpp_DIR=fftwpp_INSTALL_PREFIX/lib/cmake/fftwpp ..
  $ cmake --build . --config Release

An executable called ``example_fftwpp`` should be present in the
``build/Release`` subdirectory.
