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
   =\sum_{m=0}^M\sum_{n=0}^N
   z_{pq}\exp\Bigl[-2\mathrm{i}\pi\Bigl(\frac{mp}M+\frac{nq}N\Bigr)\Bigr]
   =\sum_{m=0}^M\sum_{n=0}^Nx^py^q\phi^{mp}\psi^{mq},

where :math:`\phi=\exp\bigl(-2\mathrm{i}\pi/M\bigr)` and
:math:`\psi=\exp\bigl(-2\mathrm{i}\pi/N\bigr)`. The above double sum can be
factored as follows

.. math::

   \hat{z}_{mn}
   =\Bigl[\sum_{m=0}^M\bigl(x\phi^p\bigr)^m\Bigr]\Bigl[\sum_{n=0}^N\bigl(y\psi^q)^n\Bigr]
   =\frac{1-\bigl(x\phi^p\bigr)^M}{1-x\phi^p}\frac{1-\bigl(y\phi^q\bigr)^N}{1-y\phi^q}.

Since :math:`\phi^M=1` and :math:`\psi^N=1`, we finally find

.. math::

   \hat{z}_{mn}=\frac{1-x^M}{1-x\phi^p}\frac{1-y^N}{1-y\phi^q}.

We will use the above formula to check the validity of our fftwpp computation.


C++ tutorial
============

The files for this example can be downloaded: C++ source and CMake file (for
compilation).


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
