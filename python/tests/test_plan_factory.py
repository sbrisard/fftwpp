import abc
import sys

import numpy as np
import pytest
import pyfftwpp as fftw

class AbstractTestPlanFactory:
    @abc.abstractmethod
    def create_plan_factory(self):
        pass

    @pytest.mark.parametrize(
        "flag_name, flag",
        [
            ("estimate", np.uint(64)),
            ("measure", np.uint(0)),
            ("patient", np.uint(32)),
            ("exhaustive", np.uint(8)),
            ("wisdom_only", np.uint(2097152)),
            ("destroy_input", np.uint(1)),
            ("preserve_input", np.uint(16)),
            ("unaligned", np.uint(2)),
        ],
    )
    def test_flag(self, flag_name, flag):
        factory = self.create_plan_factory()
        set_method = getattr(factory, f"set_{flag_name}")
        unset_method = getattr(factory, f"unset_{flag_name}")
        set_method()
        assert factory.flags == flag
        unset_method()
        assert factory.flags == 0

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
        factory = self.create_plan_factory()
        input = np.empty(ishape, dtype=factory.input_dtype)
        output = np.empty(oshape, dtype=factory.output_dtype)
        with pytest.raises(ValueError):
            factory.create_plan(input.ndim, input, output)

class TestPlanFactory_c128_c128(AbstractTestPlanFactory):
    def create_plan_factory(self):
        return fftw.PlanFactory_c128_c128()

# class TestPlanFactory_f64_c128(AbstractTestPlanFactory):
#     def create_plan_factory(self):
#         return fftw.PlanFactory_f64_c128()

# class TestPlanFactory_c128_f64(AbstractTestPlanFactory):
#     def create_plan_factory(self):
#         return fftw.PlanFactory_c128_f64()
