"""
Test of transforms per se.

In this file:

- `shape` is the logical shape of the transform
- `ishape` is the shape of the input
- `oshape` is the shape of the output
- `itype` is the type (dtype) of the input
- `otype` is the type (dtype) of the output
"""

import itertools
import numpy as np

import pytest

from pyfftwpp import PlanFactory

import faulthandler

faulthandler.enable()

shapes = [
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
]

ranks_and_shapes = [
    (1, (4, 5)),
    (1, (4, 5, 6)),
    (2, (4, 5, 6)),
    (2, (4, 5, 6, 7)),
    (3, (4, 5, 6, 7)),
    (3, (4, 5, 6, 7, 8)),
]


class AbstractPlanTest:
    def input_shape(self, rank, shape):
        raise NotImplementedError()

    def output_shape(self, rank, shape):
        raise NotImplementedError()

    def random_input(self, rank, shape):
        raise NotImplementedError()

    def fft(self, data, shape, sign=-1):
        raise NotImplementedError()

    def test_basic(self, shape, sign):
        rank = len(shape)
        data = self.random_input(rank, shape)
        act = np.empty(self.output_shape(rank, shape), dtype=self.otype)
        exp = self.fft(data, shape, sign)
        factory = PlanFactory().set_estimate()
        plan = factory.create_plan(data.ndim, data, act, sign)
        plan.execute()

        info = np.finfo(self.otype)
        np.testing.assert_allclose(act, exp, rtol=100 * info.eps, atol=1000 * info.eps)

    def test_advanced(self, rank, shape, sign):
        ndim = len(shape)
        ishape = self.input_shape(rank, shape)
        oshape = self.output_shape(rank, shape)

        # Create data, but do not initialize it, as the input might
        # get destroyed upon creation of plan
        data = np.empty(ishape, self.itype)
        act = np.zeros(oshape, self.otype)
        exp = np.zeros_like(act)

        factory = PlanFactory().set_estimate()
        plan = factory.create_plan(rank, data, act, sign)
        # Save data, which might be destroyed upon execution of the plan
        data_save = self.random_input(rank, shape)
        data[...] = data_save
        plan.execute()
        data[...] = data_save

        in1 = np.empty(ishape[:rank], self.itype)
        out1 = np.empty(oshape[:rank], self.otype)
        plan1 = factory.create_plan(rank, in1, out1, sign)

        ranges = (range(shape[i]) for i in range(rank, ndim))
        multi_indices = itertools.product(*ranges)
        for multi_index in multi_indices:
            in1[...] = data[(Ellipsis,) + multi_index]
            plan1.execute()
            exp[(Ellipsis,) + multi_index] = out1

        np.testing.assert_equal(act, exp)


class TestComplex128ToComplex128(AbstractPlanTest):
    itype = np.complex128
    otype = np.complex128

    def input_shape(self, rank, shape):
        return shape

    def output_shape(self, rank, shape):
        return shape

    def random_input(self, rank, shape):
        # Construct the dtype for the real type underlying the complex type.
        # This convoluted construct will be useful once transforms for simple
        # or quadruple precision are implemented
        rtype = self.itype(0).real.dtype
        rng = np.random.default_rng(202103120214)
        real = rng.random(size=shape, dtype=rtype)
        imag = rng.random(size=shape, dtype=rtype)
        return real + 1j * imag

    def fft(self, data, shape, sign=-1):
        if sign == -1:
            return np.fft.fftn(data)
        if sign == 1:
            return data.size * np.fft.ifftn(data)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_basic(self, shape, sign):
        super().test_basic(shape, sign)

    @pytest.mark.parametrize("rank, shape", ranks_and_shapes)
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_advanced(self, rank, shape, sign):
        super().test_advanced(rank, shape, sign)


class TestFloat64ToComplex128(AbstractPlanTest):
    itype = np.float64
    otype = np.complex128

    def input_shape(self, rank, shape):
        return shape

    def output_shape(self, rank, shape):
        oshape = list(shape)
        n = oshape[rank - 1]
        oshape[rank - 1] = n // 2 + 1
        return tuple(oshape)

    def random_input(self, rank, shape):
        rng = np.random.default_rng(202103120214)
        return rng.random(shape, self.itype)

    def fft(self, data, logical_shape, sign=-1):
        return np.fft.rfftn(data)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("sign", [-1])
    def test_basic(self, shape, sign):
        super().test_basic(shape, sign)

    @pytest.mark.parametrize("rank, shape", ranks_and_shapes)
    @pytest.mark.parametrize("sign", [-1])
    def test_advanced(self, rank, shape, sign):
        super().test_advanced(rank, shape, sign)


class TestComplex128ToFloat64(AbstractPlanTest):
    itype = np.complex128
    otype = np.float64

    def input_shape(self, rank, shape):
        ishape = list(shape)
        n = ishape[rank - 1]
        ishape[rank - 1] = n // 2 + 1
        return tuple(ishape)

    def output_shape(self, rank, shape):
        return shape

    def random_input(self, rank, shape):
        ishape = self.input_shape(rank, shape)
        oshape = self.output_shape(rank, shape)
        rng = np.random.default_rng(202103120214)
        real = np.empty(oshape, self.otype)
        complex = np.empty(ishape, self.itype)
        plan = PlanFactory().set_estimate().create_plan(rank, real, complex)
        # Define the contents of `real` only after the plan has been created,
        # since the input might be destroyed.
        real[...] = rng.random(oshape, self.otype)
        plan.execute()
        return complex

    def fft(self, data, shape, sign=-1):
        size = np.product(shape)
        out = np.fft.irfftn(data, s=shape)
        return size * out

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("sign", [-1])
    def test_basic(self, shape, sign):
        super().test_basic(shape, sign)

    @pytest.mark.parametrize("rank, shape", ranks_and_shapes)
    @pytest.mark.parametrize("sign", [-1])
    def test_advanced(self, rank, shape, sign):
        super().test_advanced(rank, shape, sign)
