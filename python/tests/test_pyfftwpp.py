import itertools
import numpy as np

import pytest

import pyfftwpp as fftw

import faulthandler

faulthandler.enable()


def test_planner_flags():
    assert fftw.PlannerFlag.measure == 0
    assert fftw.PlannerFlag.destroy_input == 1
    assert fftw.PlannerFlag.unaligned == 1 << 1
    assert fftw.PlannerFlag.exhaustive == 1 << 3
    assert fftw.PlannerFlag.preserve_input == 1 << 4
    assert fftw.PlannerFlag.patient == 1 << 5
    assert fftw.PlannerFlag.estimate == 1 << 6
    assert fftw.PlannerFlag.wisdom_only == 1 << 21


class TestPlan1d:
    dtype = np.complex128

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
        plan = fftw.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
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
        plan = fftw.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()

        exp = np.zeros_like(data)
        aux = np.zeros_like(data)
        in1 = np.zeros(shape[0], dtype=self.dtype)
        out1 = np.zeros_like(in1)
        plan1 = fftw.Plan(1, in1, out1, sign, fftw.PlannerFlag.estimate)
        for j in range(shape[1]):
            in1[:] = data[:, j]
            plan1.execute()
            aux[:, j] = out1

        in2 = np.zeros(shape[1], dtype=self.dtype)
        out2 = np.zeros_like(in2)
        plan2 = fftw.Plan(1, in2, out2, sign, fftw.PlannerFlag.estimate)
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
            ((9, 8, 7), 1)
        ],
    )
    def test_fft3(self, shape, sign):
        data = self.random(shape)

        act = np.zeros_like(data)
        plan = fftw.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()

        aux1 = np.zeros_like(data)
        in1 = np.zeros(shape[2], dtype=self.dtype)
        out1 = np.zeros_like(in1)
        plan1 = fftw.Plan(1, in1, out1, sign, fftw.PlannerFlag.estimate)
        for i in range(shape[0]):
            for j in range(shape[1]):
                in1[:] = data[i, j, :]
                plan1.execute()
                aux1[i, j, :] = out1

        aux2 = np.zeros_like(data)
        in2 = np.zeros(shape[1], dtype=self.dtype)
        out2 = np.zeros_like(in2)
        plan2 = fftw.Plan(1, in2, out2, sign, fftw.PlannerFlag.estimate)
        for i in range(shape[0]):
            for k in range(shape[2]):
                in2[:] = aux1[i, :, k]
                plan2.execute()
                aux2[i, :, k] = out2

        exp = np.zeros_like(data)
        in3 = np.zeros(shape[0], dtype=self.dtype)
        out3 = np.zeros_like(in3)
        plan3 = fftw.Plan(1, in3, out3, sign, fftw.PlannerFlag.estimate)
        for j in range(shape[1]):
            for k in range(shape[2]):
                in3[:] = aux2[:, j, k]
                plan3.execute()
                exp[:, j, k] = out3

        info = np.finfo(self.dtype)
        np.testing.assert_allclose(act, exp, rtol=100 * info.eps, atol=100 * info.eps)

    # @pytest.mark.parametrize(
    #     "ishape, oshape",
    #     [
    #         ((2,), (3,)),
    #         ((2, 3), (4,)),
    #         ((2, 3), (4, 5)),
    #         ((2, 3), (4, 3)),
    #         ((4,), (2, 3)),
    #         ((2, 4), (2, 3)),
    #         ((4, 3), (2, 3)),
    #     ],
    # )
    # def test_invalid_shape(self, ishape, oshape):
    #     input = np.empty(ishape, dtype=self.dtype)
    #     output = np.empty(oshape, dtype=self.dtype)
    #     with pytest.raises(ValueError):
    #         fftw.Plan(input.ndim, input, output, -1, fftw.PlannerFlag.estimate)
    #
    # @pytest.mark.parametrize(
    #     "rank, shape, sign",
    #     [
    #         (1, (4, 5), -1),
    #         (1, (4, 5, 6), -1),
    #         (2, (4, 5, 6), -1),
    #         (2, (4, 5, 6, 7), -1),
    #         (3, (4, 5, 6, 7), -1),
    #         (3, (4, 5, 6, 7, 8), -1),
    #     ],
    # )
    # def test_fft_advanced(self, rank, shape, sign, rtol=1e-15, atol=1e-15):
    #     ndim = len(shape)
    #     data = self.random(shape)
    #     act = np.zeros_like(data)
    #     plan = fftw.Plan(rank, data, act, sign, fftw.PlannerFlag.estimate)
    #     plan.execute()
    #
    #     in1 = np.zeros(shape[:rank], dtype=np.complex128)
    #     out1 = np.zeros_like(in1)
    #     plan1 = fftw.Plan(rank, in1, out1, sign, fftw.PlannerFlag.estimate)
    #     exp = np.zeros_like(data)
    #
    #     if rank == ndim - 1:
    #         for k in range(shape[-1]):
    #             in1[...] = data[..., k]
    #             plan1.execute()
    #             exp[..., k] = out1
    #     elif rank == ndim - 2:
    #         for h in range(shape[-2]):
    #             for k in range(shape[-1]):
    #                 in1[...] = data[..., h, k]
    #                 plan1.execute()
    #                 exp[..., h, k] = out1
    #     else:
    #         raise ValueError("unexpected rank")
    #
    #     np.testing.assert_allclose(act, exp, rtol, atol)
