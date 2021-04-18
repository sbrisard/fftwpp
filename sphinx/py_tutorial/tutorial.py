import numpy as np

import pyfftwpp

if __name__ == "__main__":
    M, N, dim = 7, 8, 2
    x = np.array([0.8, -0.9])[None, :]
    y = np.array([-1.1, 1.2])[None, :]

    m = np.arange(0, M)[:, None]
    n = np.arange(0, N)[:, None]

    φ_pow_m = np.exp(-2 * 1j * np.pi / M * m)
    ψ_pow_n = np.exp(-2 * 1j * np.pi / N * n)

    x_pow_m = x ** m
    y_pow_n = y ** n

    x_hat = (1 - x ** M) / (1 - x * φ_pow_m)
    y_hat = (1 - y ** N) / (1 - y * ψ_pow_n)

    exp = x_hat[:, None, :] * y_hat[None, :, :]

    in_ = np.empty_like(exp)
    act = np.empty_like(exp)

    factory = pyfftwpp.PlanFactory().set_estimate()
    plan = factory.create_plan(dim, in_, act, -1)
    print(f"The following plan was created:\n{plan}")

    in_[...] = x_pow_m[:, None, :] * y_pow_n[None, :, :]
    plan.execute()

    for exp_, act_ in np.nditer([exp, act]):
        print(f"expected = {exp_}, actual = {act_}")
