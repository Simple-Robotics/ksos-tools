"""This implementation is currently limited to range-only localization problems,
but can be easility extended.
Consider moving some of the code to the example classes
"""

import time
from typing import Callable

import cvxpy as cp
import numpy as np
from autograd import grad
from cert_tools.linalg_tools import rank_project
from cert_tools.sdp_solvers import solve_sdp
from scipy.optimize import minimize

DEBUG = False

TOL = 1e-10
mosek_params = {
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": TOL,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": TOL,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": TOL,
}


def get_monomial_vector(vars, degree):
    from itertools import product

    n = len(vars)
    exponents = []
    for d in range(degree + 1):
        exponents += [exp for exp in product(range(d + 1), repeat=n) if sum(exp) == d][
            ::-1
        ]
    return np.array([np.prod(vars ** np.array(exp)) for exp in exponents])


def solve_from_samples(
    samples,
    f_samples,
    basis: int | Callable = 2,
    solver: str = "MOSEK",
    orthogonalize: bool = True,
    epsilon: float = 1e-3,
    max_iters: int = 200,
    verbose: bool = False,
    linesearch: bool = False,
):
    N = samples.shape[0]

    if isinstance(basis, Callable):
        V = np.vstack([basis(sample[None, :]) for sample in samples])
    elif isinstance(basis, int):
        V = np.vstack(
            [get_monomial_vector(sample.flatten(), degree=basis) for sample in samples]
        )
    else:
        raise ValueError("monomials has to be either a callable or an integer.")

    # Test if samples are sufficient.
    W = []
    for i in range(V.shape[0]):
        w_i = np.outer(V[i], V[i])
        W.append(w_i[np.triu_indices(w_i.shape[0])][:, None])
    W = np.hstack(W)

    # check that W does not have full column rank, which means at least one column is lin. dependent.
    rank_W = np.linalg.matrix_rank(W)
    if not (W.shape[1] > rank_W):
        print("Warning: poisedness check did not pass")

    # Reduce the dimension of V by considering only the subspace.
    # See Cifuentes for more details about this.
    if orthogonalize:
        Ur, Sr, Vr = np.linalg.svd(V.T, full_matrices=False)
        T = Ur @ np.diag(Sr)
        V = Vr.T

    # Solve the new SDP problem
    info = {}
    if solver == "MOSEK":
        n = V.shape[1]
        c = cp.Variable(1)
        B = cp.Variable((n, n), symmetric=True)

        t = epsilon / n

        print("samplingSOS problem size:", B.shape)

        obj = cp.Maximize(c + t * cp.log_det(B))
        constraints = [f_samples[i] - c == V[i].T @ B @ V[i] for i in range(N)]
        constraints += [B >> 0]
        prob = cp.Problem(obj, constraints)
        try:
            t1 = time.time()
            prob.solve(verbose=verbose, accept_unknown=True, mosek_params=mosek_params)
        except cp.SolverError as e:
            print("Mosek failed:", e)
            ttot = time.time() - t1
            return None, {"cost": None, "ttot": ttot}
        else:
            ttot = time.time() - t1
            if not "optimal" in prob.status:
                print("No solution found:", prob.status)
                return None, {"cost": None, "ttot": ttot}

        y_here = np.array([constraints[i].dual_value for i in range(N)]).flatten()
        x_here = -y_here @ samples
        X_here = constraints[-1].dual_value

        if orthogonalize:
            X_here = T @ X_here @ T.T  # undo change of basis

        info.update(
            {"X": X_here, "B": B.value, "cost": c.value, "alpha": y_here, "ttot": ttot}
        )
    elif "newton" in solver:
        from ksos_tools.solvers import newton

        lambd = 1e-8

        t1 = time.time()
        if solver == "newton-features":
            problem = newton.Problem(lambd=lambd, t=epsilon / N, use_K=False)
            problem.register_fixed_samples(samples, f_samples=f_samples)
            problem.initialize_kernel_from_Phi(Phi=V.T)
            x_here, info_here = newton.damped_newton_advanced(
                problem,
                iterations=max_iters,
                verbose=verbose,
                linesearch=linesearch,
                return_B=True,
            )
        elif solver == "newton-kernel":
            problem = newton.Problem(lambd=lambd, t=epsilon / N, use_K=True)
            problem.register_fixed_samples(samples, f_samples=f_samples)
            problem.initialize_kernel_from_Phi(Phi=V.T)
            x_here, info_here = newton.damped_newton_advanced(
                problem,
                iterations=max_iters,
                verbose=verbose,
                linesearch=linesearch,
                return_B=True,
            )
        elif solver == "newton":
            problem = newton.Problem(lambd=lambd, t=epsilon / N)
            problem.register_fixed_samples(samples, f_samples=f_samples)
            problem.initialize_kernel_from_Phi(Phi=V.T)
            x_here, info_here = newton.damped_newton(
                problem, iterations=max_iters, verbose=verbose, return_B=True
            )
        ttot = time.time() - t1
        info.update(info_here)
        info["ttot"] = ttot
    return x_here, info


def solve_local(example, D, x0=None):

    if x0 is None:
        x0 = example.trajectory + np.random.normal(
            scale=0.1, size=example.trajectory.shape
        )

    fun = lambda x: example.cost_ad(D, x)

    samples = []
    f_samples = []

    def store_intermediate(xi):
        samples.append(xi)
        f_samples.append(fun(xi))

    # TODO(FD): options={"return_all": True}
    # does not allow us to access actual intermediate result,
    # only samples. That's why I created the ugly hack above with callback
    # function.
    info = minimize(
        fun,
        x0=x0.flatten(),
        jac=grad(fun),  # type: ignore
        method="BFGS",
        callback=store_intermediate,
        # options={"return_all": True},
        tol=1e-10,
    )
    info_local = {
        "samples": np.vstack(samples),
        "f_samples": np.array(f_samples),
        "cost": info.fun,
    }
    return info.x, info_local  # N x d


def solve_using_shor(example, D):
    Constraints = example.get_constraints()
    Q = example.get_cost_matrix(D=D)
    print("Shor problem size:", Q.shape)

    if DEBUG:
        x = example.get_x()
        cost = example.cost(D=D)
        assert abs(x.T @ Q @ x - cost) < 1e-10

    t1 = time.time()
    # TODO(FD): should not include compile time of MOSEK below
    X, info = solve_sdp(Q, Constraints)
    ttot = time.time() - t1
    x_hat, info_rank = rank_project(X, p=1)

    return x_hat, {
        "cost": info["cost"],
        "EVR": info_rank["EVR"],
        "H": info["H"],
        "ttot": ttot,
    }


def solve_ro_shor(dim, d, m):
    """A minimalistic implementation of Shor's relaxation for the ROsq problem.
    Might be redundant with solve_using_shor above."""
    Nm = m.shape[0]
    F = dim + 1 + 1

    A0 = np.zeros((F, F))
    A0[-1, -1] = 1

    S = np.zeros((F, F))
    S[-1, -1] = 1
    S[: dim + 1, : dim + 1] = np.eye(dim + 1)

    A = np.zeros((F, F))
    A[-2, -1] = -1 / 2
    A[-1, -2] = -1 / 2
    for i in range(dim):
        A[i, i] = 1
    A = S @ A @ S.T

    d_rs = d.reshape(-1) ** 2
    Y = m
    gamma = np.linalg.norm(Y, axis=1) ** 2
    b = d_rs - gamma

    Q = np.zeros((F, F))
    Q11 = np.zeros((dim + 1, dim + 1))
    Q11[:dim, :dim] = 4 * Y.T.dot(Y)
    Q11[dim, :dim] = -2 * Y.sum(axis=0)
    Q11[:dim, dim] = -2 * Y.sum(axis=0)
    Q11[dim, dim] = Nm

    Q[: dim + 1, : dim + 1] = Q11
    Q[-1, -1] = b.dot(b)
    Q[-1, :dim] = 2 * Y.T @ b
    Q[-1, dim] = -b.sum()
    Q[:dim, -1] = 2 * Y.T @ b
    Q[dim, -1] = -b.sum()

    F_var = cp.Variable((F, F), symmetric=True)
    constraints = [F_var >> 0]
    constraints += [cp.trace(A @ F_var) == 0]
    constraints += [cp.trace(A0 @ F_var) == 1]

    prob = cp.Problem(cp.Minimize(cp.trace(Q.T @ F_var)), constraints)  # type: ignore
    prob.solve(solver="SCS")

    assert F_var.value is not None
    return F_var.value[-1, :dim] / np.sqrt(F_var.value[-1, -1])
