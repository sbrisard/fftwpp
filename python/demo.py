import numpy as np
import pyfftwpp

if __name__ == "__main__":
    size = 8
    rng = np.random.default_rng()
    data_in = rng.random(size=size, dtype=np.float64)+1j*rng.random(size=size, dtype=np.float64)
    data_exp = np.fft.fft(data_in)
    data_out = np.zeros_like(data_in)
    plan = pyfftwpp.Plan(data_in, data_out)
    print(plan)
    print(plan.cost())
    print(plan.flops())
    plan.execute()
    print(data_in)
    print(data_out)
    print(data_exp)
