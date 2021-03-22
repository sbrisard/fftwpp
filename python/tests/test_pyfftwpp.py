import itertools
import numpy as np

import pytest

from pyfftwpp import PlanFactory

import faulthandler

faulthandler.enable()

class AbstractPlanTest:
    def random(self, shape):
        raise NotImplementedError()

    def output_shape(self, rank, input_shape):
        raise NotImplementedError()

    def fft(self, data, sign=-1):
        raise NotImplementedError()

    @pytest.mark.parametrize(
        "shape",
        [
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
            (14,),
            (15,),
            (16,),
            (31, 31),
            (31, 32),
            (32, 31),
            (32, 32),
            (7, 8, 9),
            (7, 9, 8),
            (8, 7, 9),
            (8, 9, 7),
            (9, 7, 8),
            (9, 8, 7),
        ],
    )
    def test_basic(self, shape, sign):
        data = self.random(shape)
        exp = self.fft(data, sign=sign)
        act = np.zeros_like(exp, order="C")
        factory = PlanFactory().set_estimate()
        plan = factory.create_plan(data.ndim, data, act, sign)
        plan.execute()

        info = np.finfo(self.output_type)
        np.testing.assert_allclose(act, exp, rtol=100 * info.eps, atol=1000 * info.eps)

    @pytest.mark.parametrize(
        "rank, shape",
        [
            (1, (4, 5)),
            (1, (4, 5, 6)),
            (2, (4, 5, 6)),
            (2, (4, 5, 6, 7)),
            (3, (4, 5, 6, 7)),
            (3, (4, 5, 6, 7, 8)),
        ],
    )
    def test_advanced(self, rank, shape, sign):
        ndim = len(shape)
        data = self.random(shape)
        act = np.zeros(self.output_shape(rank, shape), self.output_type)
        factory = PlanFactory().set_estimate()
        plan = factory.create_plan(rank, data, act, sign)
        plan.execute()

        exp = np.zeros_like(act)

        in1 = np.zeros(shape[:rank], dtype=self.input_type)
        out1 = np.zeros(self.output_shape(rank, in1.shape), dtype=self.output_type)
        plan1 = factory.create_plan(rank, in1, out1, sign)

        slices = tuple(slice(0, shape[i]) for i in range(rank))
        ranges = (range(shape[i]) for i in range(rank, ndim))
        multi_indices = itertools.product(*ranges)
        for multi_index in multi_indices:
            in1[...] = data[slices + multi_index]
            plan1.execute()
            exp[slices + multi_index] = out1

        np.testing.assert_equal(act, exp)


class TestComplex128ToComplex128(AbstractPlanTest):
    input_type = np.complex128
    output_type = np.complex128

    def random(self, shape):
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=shape, dtype=np.float64)
        imag = rng.random(size=shape, dtype=np.float64)
        return real + 1j * imag

    def output_shape(self, rank, input_shape):
        return input_shape

    def fft(self, data, sign=-1):
        if sign == -1:
            return np.fft.fftn(data)
        if sign == 1:
            return data.size * np.fft.ifftn(data)

    @pytest.mark.parametrize("sign", [-1, 1])
    def test_basic(self, shape, sign):
        super().test_basic(shape, sign)

    @pytest.mark.parametrize("sign", [-1, 1])
    def test_advanced(self, rank, shape, sign):
        super().test_advanced(rank, shape, sign)


class TestFloat64ToComplex128(AbstractPlanTest):
    input_type = np.float64
    output_type = np.complex128

    def fft(self, data, sign=-1):
        return np.fft.rfftn(data)

    def random(self, shape):
        rng = np.random.default_rng(202103120214)
        return rng.random(size=shape, dtype=self.input_type)

    def output_shape(self, rank, input_shape):
        shape = list(input_shape)
        n = shape[rank - 1]
        shape[rank - 1] = n // 2 + 1
        return tuple(shape)

    @pytest.mark.parametrize("sign", [-1])
    def test_basic(self, shape, sign):
        super().test_basic(shape, sign)

    @pytest.mark.parametrize("sign", [-1])
    def test_advanced(self, rank, shape, sign):
        super().test_advanced(rank, shape, sign)
