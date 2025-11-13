import numpy as np

from ksos_tools.solvers import external, newton


def cost_with_test(a, problem):
    k, N = problem.Phi.shape
    assert len(a) == N
    assert len(problem.f_samples) == N
    M = problem.get_M(a)
    test_mat = a[0] * np.outer(problem.Phi[:, 0], problem.Phi[:, 0])
    for i in range(1, N):
        test_mat += a[i] * np.outer(problem.Phi[:, i], problem.Phi[:, i])
    np.testing.assert_allclose(test_mat + problem.lambd * np.eye(k), M)
    return newton.cost_using_Phi(a, problem)


def grad_using_Phi_with_test(problem, a):
    k, N = problem.Phi.shape
    M = problem.get_M(a)
    grad_Phi = np.zeros(N)
    for i in range(N):
        grad_Phi[i] = problem.f_samples[i] - problem.t * (
            problem.Phi[:, i].T @ np.linalg.solve(M, problem.Phi[:, i])
        )

    # two ways to do the same thing
    term = np.einsum("ij,ij->j", problem.Phi, np.linalg.solve(M, problem.Phi))

    grad_Phi_here = problem.f_samples - problem.t * term
    np.testing.assert_allclose(grad_Phi, grad_Phi_here)
    return newton.grad_hess_using_Phi(problem, a)[0]


def test_gradient_finite_diff():
    lambd = 1e-3
    N = 10  # number of points
    k = 13  # number of features
    Phi = np.random.rand(k, N)
    f = np.random.rand(N)
    t = 0.1  # barrier term
    K = Phi.T @ Phi  # vectors Phi_i are rows of K! i.e. Phi[i, :]

    problem = newton.Problem(lambd=lambd, t=t)
    problem.Phi = Phi
    problem.K = K
    problem.f_samples = f

    a = np.random.rand(N)
    a /= np.sum(a)

    eps = 1e-9
    grad_Phi = grad_using_Phi_with_test(problem, a)
    grad_K, __ = newton.grad_hess_using_K(problem, a)
    for i in range(N):
        a_plus = np.copy(a)
        a_plus[i] += eps
        cost_plus = cost_with_test(a_plus, problem)
        cost_ref = cost_with_test(a, problem)
        grad_finite_diff = (cost_plus - cost_ref) / eps
        assert abs(grad_finite_diff - grad_K[i]) < 1e-3
        assert abs(grad_finite_diff - grad_Phi[i]) < 1e-3

    B_K = problem.get_B(a, use_K=True)
    B_Phi = problem.get_B(a, use_K=False)
    np.testing.assert_allclose(B_K, B_Phi, rtol=1e-3)


def test_gradients_versions():
    np.random.seed(2)

    # problem parameters
    lambd = 1e-3  # 1e-3
    N = 3  # number of points
    k = 4  # number of features
    Phi = np.random.rand(k, N) * 100 // 10
    f = np.random.rand(N) * 100 // 10
    t = 1.0  # barrier term
    K = Phi.T @ Phi  # vectors Phi_i are rows of K! i.e. Phi[i, :]

    problem = newton.Problem.create_random(N=N, k=k)

    # initialization
    a = np.random.rand(N)

    # first, compute gradient with Phi matrix
    grad_Phi, __ = newton.grad_hess_using_Phi(problem, a)

    # now, compute gradient with Kernel matrix
    K_tilde = K + lambd * np.diag(1.0 / a)
    np.testing.assert_allclose(
        K @ np.linalg.inv(K_tilde), np.linalg.solve(K_tilde, K).T
    )

    grad_K, __ = newton.grad_hess_using_K(problem, a)
    np.testing.assert_allclose(grad_K, grad_Phi)


def test_hessians_finite_diff():
    np.random.seed(1)
    lambd = 1e-3
    N = 10  # number of points
    k = 13  # number of features
    Phi = np.random.rand(k, N)
    f = np.random.rand(N)
    t = 0.1  # barrier term
    K = Phi.T @ Phi  # vectors Phi_i are rows of K! i.e. Phi[i, :]

    problem = newton.Problem(lambd=lambd, t=t)
    problem.Phi = Phi
    problem.K = K
    problem.f_samples = f

    a = np.random.rand(N)
    a /= np.sum(a)

    eps = 1e-9

    __, hess_K = newton.grad_hess_using_K(problem, a)
    __, hess_Phi = newton.grad_hess_using_Phi(problem, a=a)
    np.testing.assert_allclose(hess_K, hess_Phi, rtol=1e-4)
    for j in range(N):
        a_plus = np.copy(a)
        a_plus[j] += eps
        grad_plus, __ = newton.grad_hess_using_K(problem, a_plus)
        grad, __ = newton.grad_hess_using_K(problem, a)
        hess_row = (grad_plus - grad) / eps
        np.testing.assert_allclose(hess_K[j, :], hess_row, atol=1e-3)

        grad_plus, __ = newton.grad_hess_using_Phi(problem, a=a_plus)
        grad, __ = newton.grad_hess_using_Phi(problem, a=a)
        hess_row = (grad_plus - grad) / eps
        np.testing.assert_allclose(hess_Phi[j, :], hess_row, atol=1e-3)


def test_convergence():
    np.random.seed(1)
    lambd = 1e-3
    N = 10  # number of points
    f = np.random.rand(N)
    epsilon = 1e-8
    t = epsilon / N

    # for k=5, the kernel matrix is rank deficient.

    for k in [5, 10, 15]:
        print(f"\n\nk={k}, newton_new test using Phi\n")
        Phi = np.random.rand(k, N)
        max_iter = 100

        problem_phi = newton.Problem(
            lambd=lambd,
            use_K=False,
            t=t,
        )
        problem_phi.init_from_random(Phi, f)

        xhat, info = newton.damped_newton_advanced(
            problem_phi,
            iterations=max_iter,
            verbose=True,
        )
        assert info["success"]
        if info["B"] is None:
            print("Warning: B is None, meaning it was not psd.")

        print(f"\n\nk={k}, newton test using K\n")

        xhat, info = newton.damped_newton(
            problem_phi,
            iterations=max_iter,
            verbose=True,
        )
        assert info["success"]
        if info["B"] is None:
            print("Warning: B is None, meaning it was not psd.")

        print(f"\n\nk={k}, newton_new test using K\n")

        problem_k = newton.Problem(
            lambd=lambd,
            use_K=True,
            t=t,
        )
        problem_k.init_from_random(Phi, f)

        xhat, info = newton.damped_newton_advanced(
            problem_k,
            iterations=max_iter,
            verbose=True,
            linesearch=False,
        )
        assert info["success"]
        if info["B"] is None:
            print("Warning: B is None, meaning it was not psd.")

        print(f"\n\nk={k}, newton_new test using K with linesearch \n")
        problem_k_ls = newton.Problem(
            lambd=lambd,
            use_K=True,
            t=t,
        )
        problem_k_ls.init_from_random(Phi, f)
        xhat, info = newton.damped_newton_advanced(
            problem_k_ls,
            iterations=max_iter,
            verbose=True,
            linesearch=True,
        )
        assert info["success"]
        if info["B"] is None:
            print("Warning: B is None, meaning it was not psd.")


def test_costs():

    N = 12  # number of points
    k = 10  # number of features
    np.random.seed(1)

    # generate feasible starting points.
    a = np.random.rand(N)
    a /= np.sum(a)

    problem = newton.Problem.create_random(N=N, k=k)
    assert problem.samples is not None

    # optimize to get best cost

    z_msk, info_msk = external.solve_primal(problem, verbose=True)
    z_old, info_old = newton.damped_newton(
        problem, iterations=100, verbose=True, return_B=True
    )
    z_new, info_new = newton.damped_newton_advanced(
        problem, iterations=100, verbose=True, linesearch=True, return_B=True
    )
    assert z_new is not None
    assert z_old is not None
    assert z_msk is not None
    np.testing.assert_allclose(z_new, z_msk, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(z_old, z_msk, rtol=1e-3, atol=1e-5)

    np.testing.assert_allclose(info_new["cost"], info_msk["cost"], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(info_old["cost"], info_msk["cost"], rtol=1e-3, atol=1e-5)

    np.testing.assert_allclose(
        info_new["alpha"], info_msk["alpha"], rtol=1e-3, atol=1e-5
    )
    np.testing.assert_allclose(
        info_old["alpha"], info_msk["alpha"], rtol=1e-3, atol=1e-5
    )

    c_min_new = info_new["cost"]
    c_min_old = info_old["cost"]
    c_min_msk = info_msk["cost"]

    assert problem.f_samples is not None
    assert problem.Phi is not None
    for i, (fi, xi) in enumerate(zip(problem.f_samples, problem.samples)):
        # below must hold because we have the constraints
        # f_i - c = Phi_i^T B Phi_i for all i

        # test feasibility of obtained solution.
        cost_i_msk = fi - problem.Phi[:, i].T @ info_msk["B"] @ problem.Phi[:, i]
        np.testing.assert_allclose(cost_i_msk, c_min_msk, rtol=1e-2)
        try:
            cost_i_old = fi - problem.Phi[:, i].T @ info_old["B"] @ problem.Phi[:, i]
            np.testing.assert_allclose(cost_i_old, c_min_old, rtol=1e-2)
        except AssertionError:
            print(
                f"dampeld_newton_old: no primal feasibility at {i}: cost_i={cost_i_old:.5f}, c_min={c_min_old:.5f}"
            )
        try:
            cost_i_new = fi - problem.Phi[:, i].T @ info_new["B"] @ problem.Phi[:, i]
            np.testing.assert_allclose(cost_i_new, c_min_new, rtol=1e-2)
        except AssertionError:
            print(
                f"dampeld_newton new: no primal feasibility at {i}: cost_i={cost_i_new:3.5f}, c_min={c_min_new:.5f}"
            )


if __name__ == "__main__":
    test_costs()

    test_gradients_versions()
    test_gradient_finite_diff()
    test_hessians_finite_diff()

    test_convergence()

    print("all tests passed.")
