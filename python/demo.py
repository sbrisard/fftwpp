import numpy as np

from pyfftwpp import PlanFactory

import faulthandler
faulthandler.enable()

if __name__ == "__main__":
    input = np.zeros((9,), dtype=np.complex128)
    output = np.zeros_like(input)
    plan = PlanFactory().set_estimate().create_plan(1, input, output, -1)
    plan.execute()
