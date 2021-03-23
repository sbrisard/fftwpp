import numpy as np
import pytest

from pyfftwpp import PlanFactory


class TestPlanFactory:
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
        factory = PlanFactory()
        set_method = getattr(factory, f"set_{flag_name}")
        unset_method = getattr(factory, f"unset_{flag_name}")
        set_method()
        assert factory.flags == flag
        unset_method()
        assert factory.flags == 0

    @pytest.mark.parametrize(
        "input_type, output_type",
        [
            (np.complex128, np.complex128),
            (np.float64, np.complex128),
            (np.complex128, np.float64),
        ],
    )
    @pytest.mark.parametrize(
        "input_shape, output_shape",
        [
            ((2,), (4,)),
            ((2, 3), (4,)),
            ((2, 3), (3, 4)),
            ((2, 4), (3, 3)),
            ((2, 3), (2, 6)),
            ((4,), (2, 3)),
            ((2, 6), (2, 3)),
            ((3, 3), (2, 2)),
        ],
    )
    def test_invalid_shape(self, input_type, input_shape, output_type, output_shape):
        factory = PlanFactory()
        input = np.empty(input_shape, dtype=input_type)
        output = np.empty(output_shape, dtype=output_type)
        with pytest.raises(ValueError):
            factory.create_plan(input.ndim, input, output)
