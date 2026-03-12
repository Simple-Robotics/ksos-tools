import cvxpy as cp

from ksos_tools.utils import hvec


def find_feasible_B(alpha, c, problem, soft=False):
    """
    Find a semidefinite matrix B that satisfies the primary constraints.
    """

    K, N = problem.Phi.shape

    M = cp.Variable((K, K), PSD=True)
    # c = cp.Variable()
    if soft:
        eps = cp.Variable()

    A, b = problem.get_linear_system(alpha, c)

    if soft:
        obj = cp.Minimize(eps)
    else:
        obj = cp.Minimize(1.0)

    # constraint reduction
    C = A.T @ A
    c = A.T @ b
    constraints = [C @ hvec(M) == c]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(verbose=False)
        if M.value is not None:
            return M.value
    except Exception as e:
        print("error:", e)
        return None
    return None
