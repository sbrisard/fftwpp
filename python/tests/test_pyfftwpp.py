import itertools
import numpy as np

import pytest

import pyfftwpp as fftw

import faulthandler

faulthandler.enable()

class TestPlan1d:
    dtype = np.complex128
    Plan = fftw.Plan_c128_c128

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
        data = self.random((size,))
        if sign == -1:
            exp = np.fft.fft(data)
        else:
            exp = size * np.fft.ifft(data)
        act = np.zeros_like(data)
        plan = self.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()
        info = np.finfo(self.dtype)
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
        plan = self.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()

        exp = np.zeros_like(data)
        aux = np.zeros_like(data)
        in1 = np.zeros(shape[0], dtype=self.dtype)
        out1 = np.zeros_like(in1)
        plan1 = self.Plan(1, in1, out1, sign, fftw.PlannerFlag.estimate)
        for j in range(shape[1]):
            in1[:] = data[:, j]
            plan1.execute()
            aux[:, j] = out1

        in2 = np.zeros(shape[1], dtype=self.dtype)
        out2 = np.zeros_like(in2)
        plan2 = self.Plan(1, in2, out2, sign, fftw.PlannerFlag.estimate)
        for i in range(shape[0]):
            in2[:] = aux[i, :]
            plan2.execute()
            exp[i, :] = out2

        info = np.finfo(self.dtype)
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
        plan = self.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()

        aux1 = np.zeros_like(data)
        in1 = np.zeros(shape[2], dtype=self.dtype)
        out1 = np.zeros_like(in1)
        plan1 = self.Plan(1, in1, out1, sign, fftw.PlannerFlag.estimate)
        for i in range(shape[0]):
            for j in range(shape[1]):
                in1[:] = data[i, j, :]
                plan1.execute()
                aux1[i, j, :] = out1

        aux2 = np.zeros_like(data)
        in2 = np.zeros(shape[1], dtype=self.dtype)
        out2 = np.zeros_like(in2)
        plan2 = self.Plan(1, in2, out2, sign, fftw.PlannerFlag.estimate)
        for i in range(shape[0]):
            for k in range(shape[2]):
                in2[:] = aux1[i, :, k]
                plan2.execute()
                aux2[i, :, k] = out2

        exp = np.zeros_like(data)
        in3 = np.zeros(shape[0], dtype=self.dtype)
        out3 = np.zeros_like(in3)
        plan3 = self.Plan(1, in3, out3, sign, fftw.PlannerFlag.estimate)
        for j in range(shape[1]):
            for k in range(shape[2]):
                in3[:] = aux2[:, j, k]
                plan3.execute()
                exp[:, j, k] = out3

        info = np.finfo(self.dtype)
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
        plan = self.Plan(rank, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()

        in1 = np.zeros(shape[:rank], dtype=np.complex128)
        out1 = np.zeros_like(in1)
        plan1 = self.Plan(rank, in1, out1, sign, fftw.PlannerFlag.estimate)
        exp = np.zeros_like(data)

        slices = tuple(slice(0, shape[i]) for i in range(rank))
        ranges = (range(shape[i]) for i in range(rank, ndim))
        multi_indices = itertools.product(*ranges)
        for multi_index in multi_indices:
            in1[...] = data[slices + multi_index]
            plan1.execute()
            exp[slices+multi_index] = out1

        np.testing.assert_equal(act, exp)
