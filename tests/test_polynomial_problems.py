import os
import time

import matplotlib
import matplotlib.pylab as plt
import numpy as np

# Set matplotlib backend based on environment to avoid display issues
if os.environ.get("DISPLAY", "") == "":
    # Headless environment (e.g., CI)
    matplotlib.use("Agg")
else:
    # Local environment with GUI
    matplotlib.use("TkAgg")

from ksos_tools.examples.polynomial import Polynomial
from ksos_tools.solvers import external, ksos
from ksos_tools.solvers.helpers import find_feasible_B
from ksos_tools.solvers.problem import Problem


def test_feasible_B():
    from ksos_tools.utils import hvec

    seed = 1
    n_samples = 10
    np.random.seed(seed)
    example = Polynomial()

    # below parameters are determined by running test_surrogate_function
    kernel = "Gauss"
    sigma = 1.0
    lambd = 1e-3

    epsilon = 1e-4

    f = lambda x: example.cost(x)
    samples = example.get_valid_samples(n_samples)

    t = epsilon / n_samples
    problem = Problem(lambd, t)
    problem.register_fixed_samples(samples, f)
    problem.initialize_kernel(sigma, kernel)

    z_mosek, info_mosek = external.solve_primal(problem, solver="MOSEK", verbose=True)

    # test that B is feasible using matrix form
    assert problem.f_samples is not None
    assert problem.Phi is not None
    for i, fi in enumerate(problem.f_samples):
        err = (
            fi
            - info_mosek["cost"]
            - problem.Phi[:, i].T @ info_mosek["B"] @ problem.Phi[:, i]
        )
        assert abs(err) < 1e-5, f"constraint {i} violated by {err}"

    # test that B is feasible using linear system form
    A, b = problem.get_linear_system(info_mosek["alpha"], info_mosek["cost"])
    x = hvec(info_mosek["B"])
    np.testing.assert_allclose(A @ x, b, atol=1e-3)

    # solve feasibility problem
    B = find_feasible_B(info_mosek["alpha"], info_mosek["cost"], problem)
    x = hvec(B)
    np.testing.assert_allclose(A @ x, b, atol=1e-3)

    # note that B is not necessarily equal to B_opt!
    # np.testing.assert_allclose(B, info_mosek["B"])


def test_newton_vs_mosek():
    seed = 1
    n_samples = 6
    np.random.seed(seed)
    example = Polynomial()

    # below parameters are determined by running test_surrogate_function
    kernel = "Gauss"
    sigma = 1.0
    epsilon = 1e-4
    lambd = 1e-3

    f = lambda x: example.cost(x)
    samples = example.get_valid_samples(n_samples)

    z_dict = {}
    info_dict = {}
    for solver in ["MOSEK", "newton", "newton-kernel", "newton-features"]:
        print(f"\n\nsolving with {solver}")
        t1 = time.time()
        z_solver, info_solver = ksos.solve(
            f,
            samples.shape[1],
            solver=solver,
            lambd=lambd,
            kernel=kernel,
            sigma=sigma,
            epsilon=epsilon,
            samples=samples,
            return_B=True,
            verbose=False,
        )
        t2 = time.time() - t1
        print(f"{solver} took {t2 * 1000:.2f}ms")
        # print(f"results {solver}:\n", info_solver["alpha"])
        # print(info_solver["B"].round(2))

        z_dict[solver] = z_solver
        info_dict[solver] = info_solver

        # assert B is feasible
        assert np.all(np.linalg.eigvalsh(info_solver["B"]) >= -1e-10)

        # assert alpha is feasible
        assert abs(np.sum(info_solver["alpha"]) - 1) <= 1e-10

    for solver in ["newton", "newton-kernel", "newton-features"]:
        # make sure both find the same solution
        np.testing.assert_allclose(z_dict["MOSEK"], z_dict[solver], rtol=1e-2)
        # make sure both find the same cost
        np.testing.assert_allclose(
            info_dict["MOSEK"]["cost"], info_dict[solver]["cost"], rtol=1e-2
        )
    return


def test_surrogate_function():
    """Some tests related to the surrogate function"""
    seed = 1
    n_samples = 7
    np.random.seed(seed)

    solver = "MOSEK"

    epsilon = 1e-2
    lambd = 1e-3

    kernel_dict = {
        "Polynomial": {"params": [1, 2, 3, 4]},
        "Gauss": {"params": np.logspace(-2, 1, 4)},
        "Laplace": {"params": np.logspace(-2, 1, 4)},
    }
    for kernel, dict_values in kernel_dict.items():
        sigmas = dict_values["params"]
        example = Polynomial()

        f = lambda x: example.cost(x)
        samples = example.get_valid_samples(n_samples)

        for i, sigma in enumerate(sigmas):
            print(f"kernel: {kernel}, sigma: {sigma:.4f}")
            soft_constraints = True if kernel == "Polynomial" else False
            if kernel == "Polynomial":
                lambd_here = 0.0
            else:
                lambd_here = lambd

            theta, info = ksos.solve(
                f,
                solver=solver,
                lambd=lambd_here,
                kernel=kernel,
                sigma=sigma,
                epsilon=epsilon,
                samples=samples,
                return_B=True,
                soft_constraints=soft_constraints,
            )
            if info["B"] is None:
                print(f"  no solution!")
                continue

            # Below, an error is raised if the surrogate function does not pass through
            # the given samples (not for the Polynomial because it uses soft constraints).
            f_samples_min_c = np.array([f(sample) - info["cost"] for sample in samples])
            ksos.get_surrogate(
                info["B"],
                samples=samples,
                kernel=kernel,
                sigma=sigma,
                f_samples_min_c=f_samples_min_c,
                dx=1e-3,
                errors="raise" if kernel != "Polynomial" else "print",
            )


def test_polynomial_kernel():
    """Make sure that polynomial kernel of correct degree gives the optimal solution."""
    seed = 1
    n_samples = 7
    np.random.seed(seed)

    epsilon = 1e-5

    kernel = "Polynomial"
    sigma = 2  # large enough degree for quartic polynomial

    example = Polynomial()

    f = lambda x: example.cost(x)
    samples = example.get_valid_samples(n_samples)

    lambd_here = 1e-3

    for soft_constraints in [False, True]:
        if soft_constraints:
            solvers = ["MOSEK"]
        else:
            solvers = ["MOSEK", "newton-kernel", "newton-features"]  # disable newton

        for solver in solvers:
            print(
                f"========== solving with {solver}, soft_constraints = {soft_constraints} ============="
            )
            theta, info = ksos.solve(
                f,
                solver=solver,
                lambd=lambd_here,
                kernel=kernel,
                sigma=sigma,
                epsilon=epsilon,
                samples=samples,
                return_B=True,
                soft_constraints=soft_constraints,
                verbose=False,
            )
            assert info["B"] is not None
            assert theta is not None
            print(np.linalg.eigvalsh(info["B"]))
            print(np.linalg.eigvalsh(info["X"]))
            np.testing.assert_allclose(theta, example.x_opt, rtol=1e-2)

            f_samples_min_c = np.array([f(sample) - info["cost"] for sample in samples])
            coordinates, values = ksos.get_surrogate(
                info["B"],
                samples=samples,
                kernel=kernel,
                sigma=sigma,
                f_samples_min_c=f_samples_min_c,
                dx=1e-3,
                errors="print",
            )
            continue
            fig, ax = plt.subplots()
            example.plot(ax=ax, color="C1")
            ax.plot(coordinates[0], values, color="C1", linestyle="--")
            plt.show(block=False)
            plt.title(f"solver: {solver}, soft_constraints: {soft_constraints}")


if __name__ == "__main__":

    # import warnings
    # with warnings.catch_warnings():
    # warnings.simplefilter("error")

    # Run this test to make sure polynomial is solved exactly with polynomial kernel
    test_polynomial_kernel()

    # Run this test to make sure we have good parameters for the surrogate function.
    test_surrogate_function()

    # Test the feasible B recovery.
    test_feasible_B()

    # Run this test with good parameters to verify our solver works
    test_newton_vs_mosek()
