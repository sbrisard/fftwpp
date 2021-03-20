import itertools
import numpy as np

import pytest

import pyfftwpp as fftw

import faulthandler

faulthandler.enable()

class TestPlan1d:
    PlanFactory = fftw.PlanFactory_c128_c128

    def random(self, shape):
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=shape, dtype=np.float64)
        imag = rng.random(size=shape, dtype=np.float64)
        return real + 1j * imag

    @pytest.mark.parametrize("size, sign", itertools.product(range(2, 17), (-1, 1)))
    def test_fft(self, size, sign):
        """
        Compare 1D FFTs with direct numpy calculation. Data series are
        small enough that tolerance can be set extremely low.
        """
        factory = self.PlanFactory().set_estimate()
        data = self.random((size,))
        if sign == -1:
            exp = np.fft.fft(data)
        else:
            exp = size * np.fft.ifft(data)
        act = np.zeros_like(data)
        plan = factory.create_plan(data.ndim, data, act, sign)
        plan.execute()
        info = np.finfo(factory.output_dtype)
        np.testing.assert_allclose(act, exp, rtol=2 * info.eps, atol=2 * info.eps)

    @pytest.mark.parametrize(
        "shape, sign",
        [
            ((31, 31), -1),
            ((31, 32), -1),
            ((32, 31), -1),
            ((32, 32), -1),
            ((31, 31), 1),
            ((31, 32), 1),
            ((32, 31), 1),
            ((32, 32), 1),
        ],
    )
    def test_fft2(self, shape, sign):
        data = self.random(shape)

        act = np.zeros_like(data)
        factory = self.PlanFactory().set_estimate()
        plan = factory.create_plan(data.ndim, data, act, sign)
        plan.execute()

        exp = np.zeros_like(data)
        aux = np.zeros_like(data)
        in1 = np.zeros(shape[0], dtype=factory.input_dtype)
        out1 = np.zeros_like(in1)
        plan1 = factory.create_plan(1, in1, out1, sign)
        for j in range(shape[1]):
            in1[:] = data[:, j]
            plan1.execute()
            aux[:, j] = out1

        in2 = np.zeros(shape[1], dtype=factory.input_dtype)
        out2 = np.zeros_like(in2)
        plan2 = factory.create_plan(1, in2, out2, sign)
        for i in range(shape[0]):
            in2[:] = aux[i, :]
            plan2.execute()
            exp[i, :] = out2

        info = np.finfo(factory.output_dtype)
        np.testing.assert_allclose(act, exp, rtol=100 * info.eps, atol=100 * info.eps)

    @pytest.mark.parametrize(
        "shape, sign",
        [
            ((7, 8, 9), -1),
            ((7, 9, 8), -1),
            ((8, 7, 9), -1),
            ((8, 9, 7), -1),
            ((9, 7, 8), -1),
            ((9, 8, 7), -1),
            ((7, 8, 9), 1),
            ((7, 9, 8), 1),
            ((8, 7, 9), 1),
            ((8, 9, 7), 1),
            ((9, 7, 8), 1),
            ((9, 8, 7), 1),
        ],
    )
    def test_fft3(self, shape, sign):
        data = self.random(shape)

        act = np.zeros_like(data)
        factory = self.PlanFactory().set_estimate()
        plan = factory.create_plan(data.ndim, data, act, sign)
        plan.execute()

        aux1 = np.zeros_like(data)
        in1 = np.zeros(shape[2], dtype=factory.input_dtype)
        out1 = np.zeros_like(in1)
        plan1 = factory.create_plan(1, in1, out1, sign)
        for i in range(shape[0]):
            for j in range(shape[1]):
                in1[:] = data[i, j, :]
                plan1.execute()
                aux1[i, j, :] = out1

        aux2 = np.zeros_like(data)
        in2 = np.zeros(shape[1], dtype=factory.input_dtype)
        out2 = np.zeros_like(in2)
        plan2 = factory.create_plan(1, in2, out2, sign)
        for i in range(shape[0]):
            for k in range(shape[2]):
                in2[:] = aux1[i, :, k]
                plan2.execute()
                aux2[i, :, k] = out2

        exp = np.zeros_like(data)
        in3 = np.zeros(shape[0], dtype=factory.input_dtype)
        out3 = np.zeros_like(in3)
        plan3 = factory.create_plan(1, in3, out3, sign)
        for j in range(shape[1]):
            for k in range(shape[2]):
                in3[:] = aux2[:, j, k]
                plan3.execute()
                exp[:, j, k] = out3

        info = np.finfo(factory.output_dtype)
        np.testing.assert_allclose(act, exp, rtol=100 * info.eps, atol=100 * info.eps)

    @pytest.mark.parametrize(
        "rank, shape, sign",
        [
            (1, (4, 5), -1),
            (1, (4, 5, 6), -1),
            (2, (4, 5, 6), -1),
            (2, (4, 5, 6, 7), -1),
            (3, (4, 5, 6, 7), -1),
            (3, (4, 5, 6, 7, 8), -1),
        ],
    )
    def test_fft_advanced(self, rank, shape, sign, rtol=1e-15, atol=1e-15):
        ndim = len(shape)
        data = self.random(shape)
        act = np.zeros_like(data)
        factory = self.PlanFactory().set_estimate()
        plan = factory.create_plan(rank, data, act, sign)
        plan.execute()

        in1 = np.zeros(shape[:rank], dtype=np.complex128)
        out1 = np.zeros_like(in1)
        plan1 = factory.create_plan(rank, in1, out1, sign)
        exp = np.zeros_like(data)

        slices = tuple(slice(0, shape[i]) for i in range(rank))
        ranges = (range(shape[i]) for i in range(rank, ndim))
        multi_indices = itertools.product(*ranges)
        for multi_index in multi_indices:
            in1[...] = data[slices + multi_index]
            plan1.execute()
            exp[slices + multi_index] = out1

        np.testing.assert_equal(act, exp)
