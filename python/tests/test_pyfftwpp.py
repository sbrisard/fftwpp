import itertools
import numpy as np

import pytest

import pyfftwpp as fftw


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
    def test_fft1d(self, size, sign, rtol=1e-15, atol=1e-15):
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=size, dtype=np.float64)
        imag = rng.random(size=size, dtype=np.float64)
        data = real + 1j * imag
        if sign == -1:
            exp = np.fft.fft(data)
        else:
            exp = size * np.fft.ifft(data)
        act = np.zeros_like(data)
        plan = fftw.Plan(data, act, sign, fftw.PlannerFlag.estimate)
        plan.execute()
        np.testing.assert_allclose(act, exp, rtol, atol)

    @pytest.mark.parametrize("ishape, oshape", [((2, 3), (2,)), ((2,), (2, 3))])
    def test_fft1d_invalid_shape(self, ishape, oshape):
        input = np.empty(ishape, dtype=self.dtype)
        output = np.empty(oshape, dtype=self.dtype)
        with pytest.raises(ValueError):
            fftw.Plan(input, output, -1, fftw.PlannerFlag.estimate)

    def test_fft1d_invalid_size(self):
        input = np.empty((4,), dtype=self.dtype)
        output = np.empty((3,), dtype=self.dtype)
        with pytest.raises(ValueError):
            fftw.Plan(input, output, -1, fftw.PlannerFlag.estimate)
