import time
import warnings
from typing import Callable

import numpy as np
import scipy

from ksos_tools.solvers import external, newton
from ksos_tools.solvers.problem import LLT_METHOD, Problem, decompose, kernel_function

MAX_FAIL_COUNT = 1

TOL_GRAD = None  # 1e-8  convergence criterion for custom Newton


def solve(
    f: Callable[[np.ndarray], float],
    dim: int | None = None,
    center: np.ndarray | None = None,
    radius: np.ndarray | float | None = None,
    n_samples: int | None = None,
    samples: np.ndarray | None = None,
    f_samples: np.ndarray | None = None,
    sampling: str = "uniform",
    sampling_function: Callable[[], np.ndarray] | None = None,
    lambd: float = 1e-3,
    sigma: float = 1e-1,
    epsilon: float = 1e-3,
    decay: str | float = 0.8,
    warm_iterations: int = 1,
    return_all: bool = False,
    verbose: bool = False,
    solver: str = "newton",
    max_iters_scs: int = 10_000,
    max_iters_newton: int = 100,
    kernel: str = "Laplace",
    return_B: bool = False,
    soft_constraints: bool = False,
) -> tuple[np.ndarray | None, dict]:
    """
    Params:
    -------
    f: Callable
        The function to optimize.
    dim: int
        The dimension of the input space.
    center: np.ndarray
        The center of the input space.
    radius: np.ndarray | float
        The radius of the input space. If a float, the radius is the same for all dimensions.
    n_samples: int
        The number of samples to generate per iteration of warm restart.
    samples: np.ndarray
        If provided, the initial samples to use.
    sampling: str
        The sampling strategy to use. Either 'linspace' or 'uniform'.
        - 'linspace': samples the input space in a deterministic way
        - 'uniform': uniformly chooses random samples
    sampling_function: Callable
        If provided, the function to use for sampling.
    lambd: float
        The regularization parameter.
    sigma: float
        The kernel scaling.
    epsilon: float
        The barrier parameter.
    decay: str | float
        The decay method for the radius and sigma.
        - If a float, the radius is updated as `radius = radius * decay`, and sigma as `sigma = sigma * decay`.
        - If 'sobolev', the radius and sigma are updated using an heuristic based on the Sobolev norm.
    warm_iterations: int
        The number of warm restart iterations to perform.
    return_all: bool
        If True, returns the centers of the search space at each iteration. Otherwise, returns the last center.
    verbose: bool
        If True, prints the Sobolev norm and decay at each iteration.
    solver: str
        The solver to use. Either 'newton', 'newton-original','MOSEK', 'SCS', or 'naive'.
        - `newton`: uses the damped Newton method as suggested by Rudi et al.
        - `newton-new`: uses a new interior-point Newton method.
        - `naive`: retrieves the best sample.
        - others: uses CVXPY with the specified solver.
    max_iters_scs: int
        The maximum number of iterations for SCS.
    kernel: str
        The kernel function to use. Currently supported:
        - 'Laplace': exp(-||x-y||/sigma)
        - 'Gauss': exp(-||x-y||^2/(2sigma^2))
    return_B: bool
        If True, returns the matrix B of the solution.

    Compute the global optimum of the function f using the GloptiKernel algorithm. The approximation is done using an exponential kernel of scale sigma.
    """

    # Validate the input parameters.
    assert dim is None or dim >= 1
    if dim is None and center is not None:
        dim = len(center)
    assert center is None or len(center) == dim
    if (radius is not None) and (isinstance(radius, float) or isinstance(radius, int)):
        assert dim is not None
        radius = np.array([radius] * dim)
    assert n_samples is None or n_samples >= 1
    assert lambd >= 0
    assert (kernel == "Periodic" and isinstance(sigma, tuple) and sigma[0] > 0 and sigma[1] > 0) or sigma > 0
    assert warm_iterations >= 1
    if samples is not None:
        assert np.ndim(samples) >= 2
        if dim is not None:
            assert samples.shape[1] == dim
        else:
            dim = samples.shape[1]
        n_samples = samples.shape[0]
        center = np.mean(samples, axis=0)
        max_rad = np.max(samples - center[None, :])  # type: ignore
        min_rad = np.min(samples - center[None, :])  # type: ignore
        radius = np.max([max_rad, -min_rad])
        assert warm_iterations == 1
    assert decay == "sobolev" or (decay > 0.0 and decay < 1.0)  # type: ignore
    assert isinstance(return_all, bool)
    assert isinstance(verbose, bool)
    assert solver in [
        "newton",
        "newton-features",
        "newton-kernel",
        "MOSEK",
        "SCS",
        "naive",
    ]
    assert n_samples is not None

    info = {
        "samples": [],
        "centers": [],
        "radii": [],
        "costs": [],
        "Bs": [],
        "sigmas": [],
        "B": None,
    }
    centers = []

    if soft_constraints and solver not in ["MOSEK", "SCS"]:
        raise ValueError("Soft constraints only supported with MOSEK or SCS solver.")
    # The kernel will not be positive definite for polynomial kernel, so we must
    # use eigh there.
    llt_method = LLT_METHOD if kernel != "Polynomial" else "eigh"

    t = epsilon / n_samples
    problem = Problem(lambd=lambd, t=t)

    ttot = 0
    for iteration in range(warm_iterations):
        if verbose:
            print(
                f"it {iteration}  |  Center: {center}  |   Radius: {radius:.4f}  |  Sigma: {sigma:.2f}"
            )

        # generate samples and kernel matrix

        if samples is not None:
            if f_samples is not None:
                problem.register_fixed_samples(samples, None, f_samples)
            else:
                problem.register_fixed_samples(samples, f, None)

            if solver != "naive":
                success = problem.initialize_kernel(
                    sigma, kernel, verbose=verbose, llt_method=llt_method
                )
                if not success:
                    warnings.warn("Warning: Kernel matrix not positive definite!")
                    continue
        elif samples is None:
            fail_count = 0
            while True:
                assert center is not None
                assert radius is not None
                problem.generate_new_samples(
                    f, n_samples, center, radius, sampling, sampling_function
                )
                if solver == "naive":
                    break

                success = problem.initialize_kernel(
                    sigma, kernel, verbose=verbose, llt_method=llt_method
                )
                if success:
                    break

                fail_count += 1
                if fail_count >= MAX_FAIL_COUNT or sampling == "linspace":
                    info["cost"] = None
                    info["success"] = False
                    info["status"] = "Kernel matrix not PSD"
                    return None, info

        info["samples"].append(problem.samples)

        # ======================================================================= #
        #                     solve the optimization problem                      #
        # ======================================================================= #

        t1 = time.time()
        if solver == "naive":
            assert problem.f_samples is not None
            assert problem.samples is not None
            fmin = np.min(problem.f_samples)
            z = problem.samples[np.where(problem.f_samples == fmin)[0][0]]
            info_here = {}
            info_here["cost"] = fmin  # type: ignore
            info_here["B"] = None
        elif solver == "newton-kernel":
            problem.use_K = True
            z, info_here = newton.damped_newton_advanced(
                problem,
                iterations=max_iters_newton,
                verbose=verbose,
                return_B=return_B,
            )
        elif solver == "newton-features":
            problem.use_K = False
            z, info_here = newton.damped_newton_advanced(
                problem,
                iterations=max_iters_newton,
                verbose=verbose,
                return_B=return_B,
            )
        elif solver == "newton":
            problem.use_K = True
            z, info_here = newton.damped_newton(
                problem,
                iterations=max_iters_newton,
                verbose=verbose,
                return_B=return_B,
            )
        elif solver == "MOSEK":
            z, info_here = external.solve_primal(
                problem,
                solver=solver,
                max_iters_scs=max_iters_scs,
                soft_constraints=soft_constraints,
            )
            if info_here["status"] == "infeasible":
                warnings.warn("Infeasible problem detected!", UserWarning)
                assert isinstance(dim, int)
                return np.full(dim, np.nan), {"cost": None, "B": None}
        else:
            raise ValueError(f"Unknown solver {solver}")
        ttot += time.time() - t1
        info.update(info_here)

        # ======================================================================= #
        #                   update center and radius for next loop                #
        # ======================================================================= #
        if np.any(np.abs(z)) > 1e10:  # type: ignore
            print(f"Warning: z has very high values: {z[np.abs(z) > 1e10]}")  # type: ignore
            info["success"] = False
            info["status"] = "Newton diverged?"
            return None, info
        if np.any(z > center + radius) or np.any(z < center - radius):  # type: ignore
            print(
                "Warning: solution outside of sampling region, extrapolating! This might lead to bad solutions"
            )
            # info["success"] = True
            # return z, info
            info["status"] = "Solution extrapolated"
            center = z
        else:
            center = z

        info["costs"].append(info["cost"])
        info["Bs"].append(info["B"])
        info["centers"].append(center)
        info["radii"].append(radius)
        info["sigmas"].append(sigma)
        centers.append(center)

        if decay == "sobolev":
            assert dim == 1
            norm = sobolev_norm(samples, problem.f_samples, s=2, p=2) / np.prod(radius)  # type: ignore
            d = decay_profile(norm)
            radius = radius * d
            if verbose:
                print(f"Sobolev norm: {norm}  |  Decay: {d}  |  New radius: {radius}")
        elif iteration < warm_iterations - 1:
            assert radius is not None
            assert sigma is not None
            assert isinstance(decay, float)
            radius = radius * decay
            sigma = sigma * decay

    info["ttot"] = ttot
    info["success"] = True
    if return_all:
        return np.array(centers), info

    return center, info


def sobolev_norm(x_samples, y_samples, s=2, p=2):
    assert len(x_samples) == len(y_samples)
    y_samples = y_samples[:, 0]
    x_samples = x_samples[:, 0]

    norm = 0
    for _alpha in range(s + 1):
        # approximate the L^p norm of f^(alpha)
        # by computing a discrete approximation of the integral
        dt = np.append(np.array([x_samples[1] - x_samples[0]]), np.diff(x_samples))
        norm += np.sum(dt * np.abs(y_samples) ** p) ** (1 / p)

        # compute the next derivative
        y_samples = np.diff(y_samples) / np.diff(x_samples)
        x_samples = x_samples[:-1]
    return norm


def decay_profile(x, low=0.1, high=0.5, tau=25):
    return np.maximum(low, high - 0.3 * np.exp(-x / tau))


def get_surrogate(
    B: np.ndarray,
    samples: np.ndarray,
    sigma: float = 1e-1,
    kernel: str = "Laplacian",
    f_samples_min_c: np.ndarray | None = None,
    dx: float = 1e-2,
    errors: str = "print",
):
    """
    Interpolate kernel function between samples.

    See solve for explanation of paramters.
    """
    if samples.shape[1] == 1:
        grid_values = [np.arange(np.min(samples), np.max(samples), step=dx)[:, None]]
        evaluation_samples = grid_values[0]
    elif samples.shape[1] == 2:
        bbox_min = np.min(samples, axis=0) - dx  # d
        bbox_max = np.max(samples, axis=0) + dx  # d
        grid_values = [
            np.arange(bbox_min_d, bbox_max_d, step=dx)
            for bbox_min_d, bbox_max_d in zip(bbox_min, bbox_max)
        ]
        xx_yy_ = np.meshgrid(*grid_values, indexing="ij")
        evaluation_samples = np.hstack([c.flatten()[:, None] for c in xx_yy_])
    else:
        raise ValueError(
            f"We currently do not support dimensions > (1,2). Dimension: {samples.shape[1]}"
        )

    K_samples = np.array(
        [[kernel_function(xi, xj, sigma, kernel) for xi in samples] for xj in samples]
    )

    llt_method = LLT_METHOD if kernel != "Polynomial" else "eigh"
    R, R_inv = decompose(K_samples, method=llt_method)
    # K_pseudo_inv = Ur @ np.diag(1 / E[mask]) @ Ur.T
    # else:
    #     assert np.allclose(K_pseudo_inv, np.linalg.inv(K_samples), atol=1e-5)
    K = np.array(
        [
            [kernel_function(xi, x, sigma, kernel) for xi in samples]
            for x in evaluation_samples
        ]
    )

    if llt_method == "eigh":
        k = R_inv.T @ K.T  # type: ignore
    else:
        k = scipy.linalg.solve_triangular(R.T, K.T, lower=True)
    values = np.sum(np.multiply(k, B @ k), axis=0)

    # sanity check (this passed)
    # values2 = []
    # for i, ki in enumerate(K):
    #     # ki.T @ np.linalg.inv(R) @ B @ np.linalg.inv(R).T @ ki)
    #     if llt_method == "eigenpy":
    #         li = scipy.linalg.solve_triangular(R.T, ki, lower=True)
    #     elif llt_method == "numpy":
    #         li = scipy.linalg.solve_triangular(R.T, ki, lower=True)
    #     elif llt_method == "eigh":
    #         assert R_inv is not None
    #         li = R_inv.T @ ki
    #     values2.append(li.T @ B @ li)
    # np.testing.assert_allclose(values, values2, atol=1e-10)

    # below is for debugging purposes only
    if f_samples_min_c is not None:
        if llt_method == "eigh":
            k = R_inv.T @ K_samples.T  # type: ignore
        else:
            k = scipy.linalg.solve_triangular(R.T, K_samples.T, lower=True)
        values_samples = np.sum(np.multiply(k, B @ k), axis=0)

        for i, (fi_interp, fi) in enumerate(zip(values_samples, f_samples_min_c)):
            if abs((fi_interp - fi) / fi) > 1e-2:
                msg = f"Warning: at sample {i}, surrogate function not passing directly through f: {float(fi_interp.item()):.4f}, {float(fi.item()):.4f}"
                if errors == "print":
                    print(msg)
                elif errors == "raise":
                    raise ValueError(msg)
    return grid_values, np.array(values).reshape(*[len(g) for g in grid_values])
