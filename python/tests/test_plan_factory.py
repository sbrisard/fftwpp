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

class TestPlanFactory_c128_c128(AbstractTestPlanFactory):
    def create_plan_factory(self):
        return fftw.PlanFactory_c128_c128()

class TestPlanFactory_f64_c128(AbstractTestPlanFactory):
    def create_plan_factory(self):
        return fftw.PlanFactory_f64_c128()

class TestPlanFactory_c128_f64(AbstractTestPlanFactory):
    def create_plan_factory(self):
        return fftw.PlanFactory_c128_f64()
