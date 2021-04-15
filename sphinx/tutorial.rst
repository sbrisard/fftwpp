********
Tutorial
********

This chapter provides a tutorial to the fftwpp project. We compute the DFT of
the following 2D series of data points

.. math::

   z_{pq} = x^p y^q,\quad 0\leq p<M\quad\text{and}\quad 0\leq q<N.

The discrete Fourier transform :math:`\hat{z}_{mn}` of the above series is
explicit

.. math::

   \hat{z}_{mn}
   =\sum_{p=0}^M\sum_{q=0}^N
   z_{pq}\exp\Bigl[-2\mathrm{i}\pi\Bigl(\frac{mp}M+\frac{nq}N\Bigr)\Bigr]
   =\sum_{p=0}^M\sum_{q=0}^Nx^py^q\phi^{mp}\psi^{mq},

where :math:`\phi=\exp\bigl(-2\mathrm{i}\pi/M\bigr)` and
:math:`\psi=\exp\bigl(-2\mathrm{i}\pi/N\bigr)`. The above double sum can be
factored as follows

.. math::

   \hat{z}_{mn}
   =\Bigl[\sum_{p=0}^M\bigl(x\phi^m\bigr)^p\Bigr]\Bigl[\sum_{n=0}^N\bigl(y\psi^n)^q\Bigr]
   =\frac{1-\bigl(x\phi^m\bigr)^M}{1-x\phi^m}\frac{1-\bigl(y\psi^n\bigr)^N}{1-y\psi^n}.

Since :math:`\phi^M=1` and :math:`\psi^N=1`, we finally find

.. math::

   \hat{z}_{mn}=\frac{1-x^M}{1-x\phi^m}\frac{1-y^N}{1-y\psi^n}.

We will use the above formula to check the validity of our fftwpp computation.


C++ tutorial
============

The files for this example can be downloaded: :download:`C++
source<./cpp_tutorial/tutorial.cpp>` and :download:`CMake
file<./cpp_tutorial/CMakelists.txt>`. (for compilation).


.. _20210415083504:

Compiling the program
---------------------

``cd`` into the ``example`` subdirectory. The provided example program should be
compiled and linked against ``fftwpp``::

  $ mkdir build
  $ cd build
  $ cmake -Dfftwpp_DIR=fftwpp_INSTALL_PREFIX/lib/cmake/fftwpp ..
  $ cmake --build . --config Release

An executable called ``example_fftwpp`` should be present in the
``build/Release`` subdirectory.
