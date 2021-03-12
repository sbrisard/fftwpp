import numpy as np

import pytest

import pyfftwpp as fftw


class TestPlan1d:
    dtype = np.complex128

    @pytest.mark.parametrize("size", list(range(2, 17)))
    def test_fft1d(self, size, rtol=1e-15, atol=1e-15):
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=size, dtype=np.float64)
        imag = rng.random(size=size, dtype=np.float64)
        data = real + 1j * imag
        exp = np.fft.fft(data)
        act = np.zeros_like(data)
        plan = fftw.Plan(data, act)
        plan.execute()
        np.testing.assert_allclose(act, exp, rtol, atol)

    @pytest.mark.parametrize("ishape, oshape", [((2, 3), (2,)), ((2,), (2, 3))])
    def test_fft1d_invalid_shape(self, ishape, oshape):
        input = np.empty(ishape, dtype=self.dtype)
        output = np.empty(oshape, dtype=self.dtype)
        with pytest.raises(ValueError):
            fftw.Plan(input, output)

    def test_fft1d_invalid_size(self):
        input = np.empty((4,), dtype=self.dtype)
        output = np.empty((3,), dtype=self.dtype)
        with pytest.raises(ValueError):
            fftw.Plan(input, output)
