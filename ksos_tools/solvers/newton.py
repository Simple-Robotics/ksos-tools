import warnings

import numpy as np
import scipy
import scipy.linalg
from ksos_tools.solvers.problem import Problem

DEBUG = False

TOL_EIG_MIN = -1e-8  # values below this are considered negative.


def armijo_linesearch(cost_before, alpha, stepsize, delta, grad_H, problem: Problem):
    beta = 0.25
    gamma = 1e-4
    armijo_iter = 0
    max_armijo_iter = 50
    for armijo_iter in range(max_armijo_iter):
        alpha_trial = alpha - stepsize * delta

        if problem.use_K:
            cost_trial = cost_using_K(alpha_trial, problem)
        else:
            cost_trial = cost_using_Phi(alpha_trial, problem)

        # calculate the eigenvalues of B without calculating its inverse
        M = problem.get_M(alpha)
        eig_B = problem.t * 1 / np.linalg.eigvalsh(M)
        # verified that above is equal to np.linalg.eigvalsh(problem.get_B(alpha)), but much cheaper.

        # Eigenvalue check and Armijo condition
        if cost_trial <= cost_before + gamma * stepsize * np.dot(grad_H, delta):
            if min(eig_B) > TOL_EIG_MIN:
                return alpha_trial, {"max_iter": armijo_iter, "success": True}
            else:
                print(
                    "Warning: although cost reduced, B is not positive definite -- reducing stepsize"
                )
        stepsize *= beta
    warnings.warn("armijo failed to find a good step")
    return alpha_trial, {"max_iter": armijo_iter, "success": False}


def cost_using_Phi(a, problem):
    M = problem.get_M(a)
    det_M = np.linalg.det(M)
    if det_M <= 0:
        return np.inf
    return (
        a.T @ problem.f_samples
        - problem.t * np.log(det_M)
        + problem.t * np.log(problem.t)
        - problem.t * len(a)
    )


def cost_using_K(a, problem):
    # TODO(FD) currently deprecated because we anyways need to calculate M to get B, so might as well
    # use cost_using_Phi.

    # below we use the fact that
    # det(Phi @ np.diag(a) @ Phi.T + lambd * I) = lambda^{k-n} det(np.diag(a)) det(Phi.T @ Phi + lambd * np.diag(1/a))
    # assuming Phi is n by k, but that we have n=k
    det_arg = np.diag(a) @ (problem.K + problem.lambd * np.diag(1 / a))
    log_arg = np.prod(a) * np.linalg.det(problem.K + problem.lambd * np.diag(1 / a))

    if DEBUG:
        assert abs((np.linalg.det(det_arg) - log_arg) / log_arg) < 1e5

    if log_arg <= 0:
        return np.inf

    return (
        a.T @ problem.f_samples
        - problem.t * np.log(log_arg)
        + problem.t * np.log(problem.t)
        - problem.t * len(a)
    )


def grad_hess_using_Phi(problem, a):
    """Calcualte
    H'(a) = f - t * diag( | Phi_1' | (Phi.T @ A @ Phi + l I)^{-1} | Phi_1 ... Phi_N | )
                          |  ...   |  --------  M  --------
                          | Phi_N' |
    and H''(a) = t * (Phi.T @ M^{-1} @ Phi)^2
    """
    M = problem.get_M(a)
    if problem.Phi.shape[0] < problem.Phi.shape[1]:  # M has to be invertible
        inner = problem.Phi.T @ np.linalg.solve(M, problem.Phi)
    else:
        inner = problem.Phi.T @ np.linalg.lstsq(M, problem.Phi)[0]
    grad = problem.f_samples - problem.t * np.diag(inner)
    hess = problem.t * np.multiply(inner, inner)
    if DEBUG:
        for i in range(problem.Phi.shape[1]):  #
            Phi_i = problem.Phi[:, i]  # size: k
            grad_test = problem.f_samples[i] - problem.t * (
                Phi_i.T @ np.linalg.solve(M, Phi_i)
            )
            np.testing.assert_allclose(grad[i], grad_test, rtol=1e-5)

        for i in range(problem.Phi.shape[1]):
            for j in range(problem.Phi.shape[1]):
                Phi_i = problem.Phi[:, i]
                Phi_j = problem.Phi[:, j]
                hess_test = problem.t * (Phi_i.T @ np.linalg.lstsq(M, Phi_j)[0]) ** 2
                if hess_test > 1e-10:
                    assert abs((hess[i, j] - hess_test) / hess_test) < 1e-5
                else:
                    assert abs(hess[i, j]) < 1e-10
    return grad, hess


def grad_hess_using_K(problem, a):
    K_tilde = problem.K + problem.lambd * np.diag(1 / a)
    half_H = np.linalg.lstsq(K_tilde, problem.K)[0].T

    grad = problem.f_samples - problem.t * (1 / a) * np.diag(half_H)
    hess = problem.t * 1 / np.outer(a, a) * np.multiply(half_H, half_H.T)

    if DEBUG:
        half_H_test = problem.K @ np.linalg.inv(K_tilde)
        np.testing.assert_allclose(half_H, half_H_test, rtol=1e-3)
    return grad, hess


def damped_newton(problem, iterations=100, verbose=False, return_B=False):
    """A damped Newton method to solve the dual barrier problem

    This is an implementation of the algorithm in Section 6 of Rudi et al. 2020.

    We use the following notation:
    - Phi is a k x N matrix of features, where N is the number of samples.
    - K = Phi.T @ Phi is the NxN kernel matrix.
    - M = Phi @ Phi.T is the kxk moment matrix.

    The goal is to solve the following optimization problem:

    ..math::
        max c - lambda * trace(B) + t log det (B)
            s.t. f_i - Phi_i^T B Phi_i >= c, i=1,...,N
                 B >= 0

    where
    - Phi_i are the columns of Phi, and B is a kxk psd matrix
    - f_i are samples of a function f at points x_i, i=1, ..., N
    """
    assert problem.nu == 0, "Not implemented for nu > 0."
    n = len(problem.f_samples)

    assert (
        len(problem.f_samples) == n
    ), "The number of samples and the number of function values must be equal."

    def H_prime(alpha, C):
        return problem.f_samples - problem.t / alpha * np.diag(C)

    def H_pprime(alpha, C):
        return problem.t / np.outer(alpha, alpha) * np.multiply(C, C.T)

    def solve_newton_system(alpha):
        # C is the term K (K + lambd * Diag(a)^-1)^-1
        K_tilde = problem.K + problem.lambd * np.diag(1 / alpha)
        C = np.linalg.lstsq(K_tilde, problem.K)[0].T
        H_p = H_prime(alpha, C)
        H_pp = H_pprime(alpha, C)

        try:
            # llt = eigenpy.LLT(H_pp)
            # solve = lambda x: llt.solve(x)
            a, b = scipy.linalg.cho_factor(H_pp)
            solve = lambda x: scipy.linalg.cho_solve((a, b), x)
        except scipy.linalg.LinAlgError:
            warnings.warn("Hessian is not positive definite?")
            a, b = scipy.linalg.lu_factor(H_pp)
            solve = lambda x: scipy.linalg.lu_solve((a, b), x)
        den = solve(np.ones(n))
        num = solve(H_p)
        c = num.sum() / den.sum()
        Delta = num - c * den
        # Delta = solve((a, b), H_p - num / den * np.ones(n))
        return Delta, c, Delta.T @ H_pp @ Delta

    alpha = np.ones(n) / n

    success = False
    for i in range(iterations):
        # update alpha
        Delta, c, lambda_alpha_sq = solve_newton_system(alpha)
        stepsize = 1 / (1 + np.sqrt(1 / problem.t * lambda_alpha_sq))
        alpha = alpha - stepsize * Delta
        if lambda_alpha_sq < 0:
            warnings.warn(
                f"Warning: Newton decrement is negative, meaning Hessian was not positive definite!"
            )
            status = "Hessian not positive definite"
            success = False
            break

        if lambda_alpha_sq < problem.epsilon:
            status = "converged in Newton decrement."
            success = True
            break

        if verbose:
            B = problem.get_B(alpha)
            eig_min_after = np.linalg.eigvalsh(B).min()

            C = np.linalg.solve(
                problem.K + problem.lambd * np.diag(1 / alpha), problem.K
            ).T
            H_pp = H_pprime(alpha, C)
            cond_H = np.linalg.cond(H_pp)
            z_hat = (alpha[:, None] * problem.samples).sum(axis=0)
            msg = f"it{i:3.0f}, l(a)^2: {lambda_alpha_sq:.3e}, c: {c:.5f}, min_eig: {eig_min_after:.3e}, cond(H): {cond_H:.3e}"
            msg += f" {alpha.round(3)}"
            print(msg)
            # print(f"it{i:3.0f}, Newton decrement: {lambda_dec:.3e}")

    if i == iterations - 1:
        success = False
        status = f"did not converge in {iterations} iterations."

    z_hat = (alpha[:, None] * problem.samples).sum(axis=0)

    # Problem: if there is not a unique solution for B, then we are not sure if either
    # of the below actually gives a feasible solution to phi_i'B'phi_i = f_i - c.
    # That's why there is the find_feasible_B function below. But it is not behaving very
    # stably yet and it is costly, so for now we are ignoring it.
    B = None
    X = None
    if return_B:
        B = problem.get_B(alpha)
        X = problem.get_M(alpha)
        # B = find_feasible_B(alpha, c, problem)

    info = {
        "cost": c,
        "alpha": alpha,
        "status": status,
        "B": B,
        "X": X,
        "success": success,
        "alpha": alpha,
    }
    return z_hat, info


def damped_newton_advanced(
    problem: Problem, iterations=100, verbose=False, linesearch=False, return_B=False
):
    """Evolution of the damped_newton method exposing more functionalities such as
    using kernelized vs. feature version, and doing Armijo backtracking line search.
    """
    assert problem.f_samples is not None
    assert problem.samples is not None
    assert problem.Phi is not None

    if problem.use_K and problem.K is None:
        problem.K = problem.Phi.T @ problem.Phi
    elif problem.use_K and problem.K is not None:
        np.testing.assert_almost_equal(
            problem.K, problem.Phi.T @ problem.Phi, decimal=5
        )

    n = len(problem.f_samples)
    assert (
        len(problem.f_samples) == n
    ), "The number of samples and the number of function values must be equal."

    alpha = (
        np.ones(n) / n
    )  # feasible starting point dual variables (they have to sum to 1)
    c = np.inf  # feasible starting point cost

    def solve_newton_system(alpha, grad_H, hess_H):
        ones = np.ones((len(alpha), 1))

        # Below is an alternative way to solve, but empirically not better or worse.
        # kkt_matrix = np.block(
        #     [
        #         [hess_H, ones],
        #         [ones.T, np.zeros((1, 1))],
        #     ]
        # )
        # delta_gamma = np.linalg.lstsq(kkt_matrix, np.hstack([grad_H, 1 - alpha.sum()]))[
        #     0
        # ]
        # delta_new = delta_gamma[:-1]
        # c_new = -delta_gamma[-1]
        # np.testing.assert_allclose(delta_new1, delta_new, rtol=1e-5)
        # np.testing.assert_allclose(c_new1, c_new, rtol=1e-5)

        # note that ones.T @ A is equivalent to A.sum(axis=0)
        try:
            a, b = scipy.linalg.cho_factor(hess_H)
            solve = lambda x: scipy.linalg.cho_solve((a, b), x)
            # llt = eigenpy.LLT(hess_H)
            # solve = lambda x: llt.solve(x)
        except scipy.linalg.LinAlgError:
            a, b = scipy.linalg.lu_factor(hess_H)
            solve = lambda x: scipy.linalg.lu_solve((a, b), x)
        g1 = solve(grad_H).flatten()
        g2 = solve(ones).flatten()
        c_new = g1.sum() / g2.sum()
        delta_new = g1 - c_new * g2

        # solve((a, b), grad_H + np.full(len(alpha), c_new))[0]
        lambda_alpha = delta_new.T @ hess_H @ delta_new
        if DEBUG and lambda_alpha > 1e-5:
            lambda_alpha_test = delta_new.T @ grad_H
            if lambda_alpha > 1e3:
                assert (
                    abs(lambda_alpha - lambda_alpha_test) / lambda_alpha <= 1e-5
                ), f"two values don't match: {lambda_alpha} != {lambda_alpha_test}"
            else:
                assert (
                    abs(lambda_alpha - lambda_alpha_test) <= 1e-1
                ), f"two values don't match: {lambda_alpha} != {lambda_alpha_test}"
        return delta_new, c_new, lambda_alpha

    success = False
    delta = None
    for i in range(iterations):
        # construct gradient and Hessian using the current alpha.
        assert problem.Phi is not None

        M = None
        if problem.use_K:
            grad_H, hess_H = grad_hess_using_K(problem, alpha)
        else:
            grad_H, hess_H = grad_hess_using_Phi(problem, alpha)

        delta, c, lambda_alpha_sq = solve_newton_system(alpha, grad_H, hess_H)

        # double check that the Newton system sovled correctly
        dual_residual = hess_H @ delta + c - grad_H
        primal_residual = delta.sum() + (1 - alpha.sum())
        try:
            max_res = np.max(np.abs(dual_residual))
            assert max_res <= 1e-1
        except Exception as e:
            warnings.warn(f"dual residual too large: {max_res} > 1e-3")
            break

        try:
            max_res = np.max(np.abs(primal_residual))
            assert max_res <= 1e-1
        except Exception as e:
            warnings.warn(f"primal residual too large: {max_res} > 1e-3")
            break

        if lambda_alpha_sq < 0:
            warnings.warn(
                f"Warning: Newton decrement is negative, meaning Hessian was not positive definite!"
            )
            status = "Hessian not positive definite"
            success = False
            break

        stepsize = 1 / (1 + np.sqrt(1 / problem.t * lambda_alpha_sq))

        # Armijo backtracking line search
        if linesearch or DEBUG:
            if problem.use_K:
                cost_before = cost_using_K(alpha, problem)
            else:
                cost_before = cost_using_Phi(alpha, problem)

        if linesearch:
            alpha, info_armijo = armijo_linesearch(
                cost_before, alpha, stepsize, delta, grad_H, problem
            )
        else:
            alpha = alpha - stepsize * delta

        if lambda_alpha_sq < problem.epsilon:
            status = f"converged in Newton decrement after {i} iterations."
            success = True
            break

        if DEBUG:
            if problem.use_K:
                cost_after = cost_using_K(alpha, problem)
            else:
                cost_after = cost_using_Phi(alpha, problem)
            if cost_after > cost_before:
                msg = "Error: cost_after > cost_before, turn on linesearch to prevent this."
                if linesearch:
                    raise ValueError(msg)
                else:
                    print(msg)

        if DEBUG:
            B = problem.get_B(alpha)

            # check dual feasibility.
            # residuals = problem.f_samples - np.einsum("ik,ij,jk->k", problem.Phi, B, problem.Phi)
            residuals = problem.f_samples - np.diag(problem.Phi.T @ B @ problem.Phi)
            print("residuals (should all be constant):", residuals)

        if verbose:
            # To get the eigenvalues of B, we calculate the eigenvalues of B_inv and then take the
            # reciprocal. That's faster than doign the inverse each time.
            # TODO(FD) we are calculating M here, we don't need to calcualte it again in the next round!
            B_inv = 1 / problem.t * problem.get_M(alpha)
            eig_min_after = 1 / np.linalg.eigvalsh(B_inv).max()
            cond_H = np.linalg.cond(hess_H)

            # assert abs(np.sum(delta)) < 1e-2
            msg = f"it{i:3.0f}, l(a)^2: {lambda_alpha_sq:.3e}, c: {c:.5f}, min_eig: {eig_min_after:.3e}, cond(H): {cond_H:.3e}"
            msg += f" {alpha.round(3)}"
            if linesearch:
                msg += f", armijo iter: {info_armijo['max_iter']}"
            print(msg)

    if not success:
        status = f"did not converge in {iterations} iteratioons."

    z_hat = (alpha[:, None] * problem.samples).sum(axis=0)

    # calcualte the optimal solution
    # TODO(FD) recalculating M in below's function
    B = None
    X = None
    if return_B:
        X = problem.get_M(alpha)
        B = problem.get_B(alpha)
        eigs = np.linalg.eigvalsh(B)
        if any(eigs < TOL_EIG_MIN):
            warnings.warn("B is not positive!", UserWarning)
    info = {
        "cost": c,
        "alpha": alpha,
        "status": status,
        "success": success,
        "B": B,
        "X": X,
    }
    return z_hat, info
