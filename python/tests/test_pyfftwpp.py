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

    @pytest.mark.parametrize("size, sign", itertools.product(range(2, 17), (-1, 1)))
    def test_fft(self, size, sign):
        """
        Compare 1D FFTs with direct numpy calculation. Data series are
        small enough that tolerance can be set extremely low.
        """
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=size, dtype=np.float64)
        imag = rng.random(size=size, dtype=np.float64)
        data = real + 1j * imag
        if sign == -1:
            exp = np.fft.fft(data)
        else:
            exp = size * np.fft.ifft(data)
        act = np.zeros_like(data)
        plan = fftw.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()
        info = np.finfo(self.dtype)
        np.testing.assert_allclose(act, exp, rtol=2 * info.eps, atol=2 * info.eps)

    @pytest.mark.parametrize("shape, sign", [((4, 5), -1)])
    def test_fft2(self, shape, sign, rtol=1e-15, atol=1e-15):
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=shape, dtype=np.float64)
        imag = rng.random(size=shape, dtype=np.float64)
        data = real + 1j * imag
        if sign == -1:
            exp = np.fft.fft2(data)
        else:
            exp = data.size * np.fft.ifft2(data)
        act = np.zeros_like(data)
        plan = fftw.Plan(data.ndim, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()
        np.testing.assert_allclose(act, exp, rtol, atol)

    @pytest.mark.parametrize(
        "ishape, oshape",
        [
            ((2,), (3,)),
            ((2, 3), (4,)),
            ((2, 3), (4, 5)),
            ((2, 3), (4, 3)),
            ((4,), (2, 3)),
            ((2, 4), (2, 3)),
            ((4, 3), (2, 3)),
        ],
    )
    def test_invalid_shape(self, ishape, oshape):
        input = np.empty(ishape, dtype=self.dtype)
        output = np.empty(oshape, dtype=self.dtype)
        with pytest.raises(ValueError):
            fftw.Plan(input.ndim, input, output, -1, fftw.PlannerFlag.estimate)

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
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=shape, dtype=np.float64)
        imag = rng.random(size=shape, dtype=np.float64)
        data = real + 1j * imag
        act = np.zeros_like(data)
        plan = fftw.Plan(rank, data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()

        in1 = np.zeros(shape[:rank], dtype=np.complex128)
        out1 = np.zeros_like(in1)
        plan1 = fftw.Plan(rank, in1, out1, sign, fftw.PlannerFlag.estimate)
        exp = np.zeros_like(data)

        if rank == ndim - 1:
            for k in range(shape[-1]):
                in1[...] = data[..., k]
                plan1.execute()
                exp[..., k] = out1
        elif rank == ndim - 2:
            for h in range(shape[-2]):
                for k in range(shape[-1]):
                    in1[...] = data[..., h, k]
                    plan1.execute()
                    exp[..., h, k] = out1
        else:
            raise ValueError("unexpected rank")

        np.testing.assert_allclose(act, exp, rtol, atol)
