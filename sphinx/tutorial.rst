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
   :label: eq20210418065727

   \hat{z}_{mn}=\frac{1-x^M}{1-x\phi^m}\frac{1-y^N}{1-y\psi^n}.

We will use the above formula to check the validity of our fftwpp
computation. To make things a little more interesting, we will consider two
pairs of :math:`(x, y)` values at a time.

In both the C++ and Python versions, computing the above DFT reduces to the
following steps

1. Create the input and output arrays.
2. Create a :cpp:class:`fftwpp::PlanFactory` or :py:class:`pyfftwpp.PlanFactory`.
3. Set the various `planner flags`_ with the ``set/unset`` methods.
4. Create the :cpp:class:`fftwpp::Plan` or :py:class:`pyfftwpp.Plan`.
5. Set the values of the input data.
6. Execute the :cpp:class:`fftwpp::Plan` or :py:class:`pyfftwpp.Plan`.

.. note:: The values in the input array should be set only *after* creation of
          the ``Plan``. Indeed (quoting from the FFTW documentation): “these
	  arrays are overwritten during planning, unless ``FFTW_ESTIMATE`` is
	  used in the flags. (The arrays need not be initialized, but they must
	  be allocated.)”

In the remainder of this chapter, we first present the :ref:`C++ implementation
<sec20210418121122>`, then the :ref:`Python implementation <sec20210418120954>`
of this example.


.. _sec20210418121122:

C++ tutorial
============

The files for this example can be downloaded: :download:`C++
source<./cpp_tutorial/tutorial.cpp>` and :download:`CMake
file<./cpp_tutorial/CMakelists.txt>` (for compilation). Note that this tutorial
uses C++20 features.

The program is quite verbose, but the really interesting parts reduce to a few
line, so keep reading!

We start with a few includes and type definitions.

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :end-before: end20210418063811

The following function returns a 2d array (row-major order) of the powers of the
specified 1d array, ``x``. More specifically, ``x_pow[k, i] = x[i] ** k``.

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418063940
   :end-before: end20210418063940

The following function returns a 1d array of the powers of the complex with unit
modulus ``exp(1j * arg)``.

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418064253
   :end-before: end20210418064253

The following function returns the expected 1d DFT of the powers of the array
`x`. More precisely, given :math:`\bigl(x_1, \ldots, x_d)`, the function
computes

.. math::

   \hat{x}_{mk}=\frac{1-x_k^M}{1-x_k\phi^m}.

The returned array, ``x_hat``, is such that :math:`\hat{x}_{mk}` is stored in
``x_hat[m, k]`` (in row-major order).

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418064539
   :end-before: end20210418064539

We are now ready for the tutorial itself. The ``tutorial`` function takes
``x``, ``y``, ``M`` and ``N`` as input. The size of the ``x`` and ``y`` arrays
is specified as a template parameter. We first compute the powers of ``x`` and
``y`` and the corresponding 1d DFTs, in order to later compute the expected 2d
DFT.

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418065331
   :end-before: end20210418065331

Then, we define the various arrays (**step 1**): ``in`` stores the input data,
``exp`` is the *expected* data [computed through
Eq. :math:numref:`eq20210418065727`] and ``act`` is the *actual* DFT (computed
with FFTW). These are 3d arrays of shape ``(M, N, 2)``: ``in[m, n, k]`` holds
the value of :math:`x_k^my_k^n`, while ``exp[m, n, k]`` and ``act[m, n, k]``
hold the expected and actual values of :math:`\hat{x}_{mk}\hat{y}_{nk}`.

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418065501
   :end-before: end20210418065501

We then create a :cpp:class:`fftwpp::PlanFactory` instance (**step 2**) and set
the planner flags (**step 3**).

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418065917
   :end-before: end20210418065917

The factory is used to create an instance of :cpp:class:`fftwpp::Plan` (**step
4**). Note that the first parameter, ``2`` in the present case, means that the
DFT will be computed along the first 2 axes only. In other words, ``act[...,
k]`` holds the 2d DFT of ``in[..., k]`` for ``k = 0, 1``.

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418075847
   :end-before: end20210418075847

Now that the :cpp:class:`fftwpp::Plan` has been created, the values of the input
array can be set (**step 5**). We take advantage of the same loop to also
compute the expected values of the DFT.

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418080158
   :end-before: end20210418080158

The :cpp:class:`fftwpp::Plan` is now executed to compute the transform (**step
6**).

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418080316
   :end-before: end20210418080316

Finally, the expected and actual outputs are printed out

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418202159
   :end-before: end20210418202159

The ``tutorial()`` function is then called in the ``main()`` function as follows

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418080545
   :end-before: end20210418080545

To sum up, create a :cpp:class:`fftwpp::Factory` and set the planner flags

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418065917
   :end-before: end20210418065917

Then create the :cpp:class:`fftwpp::Plan`

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418075847
   :end-before: end20210418075847

Finally set the data and execute the :cpp:class:`fftwpp::Plan`

.. literalinclude:: cpp_tutorial/tutorial.cpp
   :language: cpp
   :start-after: begin20210418080316
   :end-before: end20210418080316


.. _20210415083504:

Compiling and running the program
---------------------------------

``cd`` into the ``sphinx/cpp_tutorial`` subdirectory. Then issue the following
commands::

  $ mkdir build
  $ cd build
  $ cmake -Dfftwpp_DIR=fftwpp_INSTALL_PREFIX/lib/cmake/fftwpp ..
  $ cmake --build . --config Release

An executable called ``tutorial`` should be present in the ``build/Release``
subdirectory. Running this executable results in the following output

.. code-block:: none

   The following plan was created plan: (dft-rank>=2/1
     (dft-vrank>=1-x7/1
       (dft-direct-8-x2 "n1_8"))
     (dft-direct-7-x16 "n1_7"))

   expected = (-2.15181,0), actual = (-2.15181,0)
   expected = (12.8371,0), actual = (12.8371,0)
   expected = (-2.1334,-0.93339), actual = (-2.1334,-0.93339)
   ..........................................................
   expected = (-0.514645,-0.269777), actual = (-0.514645,-0.269777)
   expected = (-1.86644,-2.72753), actual = (-1.86644,-2.72753)


.. _sec20210418120954:

Python tutorial
===============

As expected, the Python code is more compact. We first import the required
modules and define the parameters of the tutorial.

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418181255
   :end-before: end20210418181255

We then compute the powers of :math:`\phi`, :math:`\psi`, :math:`x` and
:math:`y`, as well as :math:`\hat{x}` and :math:`\hat{y}`, and finally the
expected data.

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418181632
   :end-before: end20210418181632

The other arrays, (input data and its actual discrete Fourier transform) have
the same shape as ``exp`` (expected discrete Fourier transform). They are
created accordingly (**step 1**).

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418181818
   :end-before: end20210418181818

We then create a :py:class:`pyfftwpp::PlanFactory` instance (**step 2**) and set
the planner flags (**step 3**).

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418182444
   :end-before: end20210418182444

The factory is used to create an instance of :py:class:`pyfftwpp.Plan` (**step
4**). Note that the first parameter, ``2`` in the present case, means that the
DFT will be computed along the first 2 axes only. In other words, ``act[...,
k]`` holds the 2d DFT of ``in_[..., k]`` for ``k = 0, 1``.

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418182614
   :end-before: end20210418182614

Now that the :py:class:`pyfftwpp.Plan` has been created, the values of the input
array can be set (**step 5**).

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418183338
   :end-before: end20210418183338

The :py:class:`pyfftwpp::Plan` is now executed to compute the transform (**step
6**).

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418183425
   :end-before: end20210418183425

Finally, the expected and actual outputs are printed out

.. literalinclude:: py_tutorial/tutorial.py
   :language: python
   :start-after: begin20210418202605
   :end-before: end20210418202605

Upon execution, the above script produces the following output

.. code-block:: none

    The following plan was created:
    (dft-rank>=2/1
      (dft-vrank>=1-x7/1
        (dft-direct-8-x2 "n1_8"))
      (dft-direct-7-x16 "n1_7"))
    expected = (-2.151811557126403+0j), actual = (-2.151811557126404+0j)
    expected = (12.837129427724797+0j), actual = (12.837129427724797+0j)
    expected = (-2.1334009536144922-0.9333897024884867j), actual = (-2.1334009536144922-0.933389702488487j)
    .......................................................................................................
    expected = (-0.5146446886147904-0.2697769241798792j), actual = (-0.5146446886147907-0.26977692417987936j)
    expected = (-1.8664372836457583-2.7275262160500926j), actual = (-1.8664372836457572-2.7275262160500935j)

To sum up, create a :py:class:`pyfftwpp::Factory` and set the planner flags

.. literalinclude:: py_tutorial/tutorial.py
   :language: py
   :start-after: begin20210418182444
   :end-before: end20210418182444

Then create the :py:class:`pyfftwpp::Plan`

.. literalinclude:: py_tutorial/tutorial.py
   :language: py
   :start-after: begin20210418182614
   :end-before: end20210418182614

Finally, set the data and execute the :py:class:`pyfftwpp::Plan`

.. literalinclude:: py_tutorial/tutorial.py
   :language: py
   :start-after: begin20210418183425
   :end-before: end20210418183425


.. _planner flags: http://fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags
