# begin20210418181255
import numpy as np

import pyfftwpp

if __name__ == "__main__":
    M, N, dim = 7, 8, 2
    x = np.array([0.8, -0.9])[None, :]
    y = np.array([-1.1, 1.2])[None, :]
    # end20210418181255

    #begin20210418181632
    m = np.arange(0, M)[:, None]
    n = np.arange(0, N)[:, None]

    φ_pow_m = np.exp(-2 * 1j * np.pi / M * m)
    ψ_pow_n = np.exp(-2 * 1j * np.pi / N * n)

    x_pow_m = x ** m
    y_pow_n = y ** n

    x_hat = (1 - x ** M) / (1 - x * φ_pow_m)
    y_hat = (1 - y ** N) / (1 - y * ψ_pow_n)

    exp = x_hat[:, None, :] * y_hat[None, :, :]
    # end20210418181632

    # begin20210418181818
    in_ = np.empty_like(exp)
    act = np.empty_like(exp)
    #end20210418181818

    # begin20210418182444
    factory = pyfftwpp.PlanFactory().set_estimate()
    # end20210418182444

    # begin20210418182614
    plan = factory.create_plan(2, in_, act, -1)
    # end20210418182614

    print(f"The following plan was created:\n{plan}")

    # begin20210418183338
    in_[...] = x_pow_m[:, None, :] * y_pow_n[None, :, :]
    # end20210418183338

    #begin20210418183425
    plan.execute()
    #end20210418183425

    #begin20210418202605
    for exp_, act_ in np.nditer([exp, act]):
        print(f"expected = {exp_}, actual = {act_}")
    #end20210418202605
