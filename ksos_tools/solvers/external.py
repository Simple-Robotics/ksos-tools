import warnings

import cvxpy as cp
import numpy as np

DEBUG = False


def solve_primal(
    problem,
    solver="MOSEK",
    max_iters_scs=10_000,
    verbose=False,
    soft_constraints=False,
    mu=1e3,
):
    # Build the optimization problem
    n = problem.Phi.shape[0]
    B = cp.Variable((n, n), symmetric=True)
    c = cp.Variable(1)

    constraints = [B >> 0]
    if soft_constraints:
        residual = problem.f_samples - c - cp.diag(problem.Phi.T @ B @ problem.Phi)
        obj = cp.Maximize(
            c
            - problem.lambd * cp.trace(B)
            + problem.t * cp.log_det(B)
            - mu * cp.sum_squares(residual)
        )
    else:
        obj = cp.Maximize(c - problem.lambd * cp.trace(B) + problem.t * cp.log_det(B))
        constraints += [
            problem.f_samples[j] - c == problem.Phi[:, j].T @ B @ problem.Phi[:, j]
            for j in range(problem.n_samples)
        ]

    # Solve the optimization problem
    prob = cp.Problem(
        obj,
        constraints,  # type: ignore
    )
    if solver == "MOSEK":
        prob.solve(solver, accept_unknown=True, verbose=verbose)
    elif solver == "SCS":
        prob.solve(solver, max_iters=max_iters_scs, verbose=verbose)

    info = {
        "cost": c.value,
        "B": B.value,
        "X": constraints[0].dual_value,
        "status": prob.status,
        "success": prob.status == "optimal",
    }
    z = None
    if B.value is not None:
        if soft_constraints:
            alpha = -2 * mu * residual.value
        else:
            alpha = np.array(
                [-c.dual_value[0] for c in constraints[-problem.n_samples :]]  # type: ignore
            )  # n_samples x 1
            assert abs(np.sum(alpha) - 1) < 1e-2
        info["alpha"] = alpha
        z = (alpha[:, None] * problem.samples).sum(axis=0)

    # some sanity checks
    if DEBUG and B.value is not None:
        for i in range(len(problem.f_samples)):
            lhs = problem.f_samples[i] - c.value
            err = abs(lhs - problem.Phi[:, i].T @ B.value @ problem.Phi[:, i])
            if err < 1e-5:
                continue
            err_rel = abs(err / lhs)
            if err_rel > 1e-2:
                warnings.warn(
                    f"Large feasibility error at {i}: absolute {err}, relative {err_rel}"
                )
    return z, info
