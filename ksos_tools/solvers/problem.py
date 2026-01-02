import warnings
from itertools import combinations_with_replacement
from typing import Callable

import eigenpy
import numpy as np

from ksos_tools.utils import duplication_matrix, get_samples, hvec

LLT_METHOD = "eigenpy"  # "numpy"  # "eigh"  # "numpy"  # "numpy" or "eigenpy"
# LLT_METHOD = "eigh"


def generate_monomial_exponents_of_degree(n_vars, degree):
    """Generate all exponent tuples for monomials of up to degree in n_vars variables."""
    exponents = []
    for total_degree in range(degree + 1):
        for c in combinations_with_replacement(range(n_vars), total_degree):
            exp = [0] * n_vars
            for idx in c:
                exp[idx] += 1
            exponents.append(tuple(exp))
    return exponents


def get_monomial_vectors(degree: int, samples: np.ndarray):
    n_vars = samples.shape[1]
    # generate all monomial exponents of given degree

    exponents = generate_monomial_exponents_of_degree(n_vars, degree)
    k = len(exponents)
    N = samples.shape[0]
    Phi = np.zeros((k, N))
    for i, exp_tuple in enumerate(exponents):
        Phi[i, :] = np.prod(samples ** np.array(exp_tuple)[None, :], axis=1)
    return Phi


def kernel_function(x, y, sigma, kernel):
    if kernel == "Laplace":
        return np.exp(-np.linalg.norm(x - y) / sigma)
    elif kernel == "Gauss":
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma**2))
    elif kernel == "Polynomial":
        return (1 + np.inner(x, y)) ** int(sigma)
    elif kernel == "Periodic":
        p, l = sigma
        return np.prod(np.exp(-2 * (np.sin(np.pi * (x - y) / p) ** 2) / l**2))
    else:
        raise ValueError(f"unknown kernel function {kernel}")


def decompose(K, method):
    R_inv = None
    if method == "eigh":
        E, U = np.linalg.eigh(K)
        mask = E > 1e-10
        rank = mask.sum()
        if rank < K.shape[0]:
            warnings.warn(
                f"Kernel matrix not full-rank; using rank: {rank}<{K.shape[0]}"
            )
        R = (E[mask] ** 0.5 * U[:, mask]).T
        R_inv = R.T @ np.diag(1.0 / E[mask])  # r x N
        np.testing.assert_allclose(R @ R_inv, np.eye(rank), atol=1e-8)
    elif method == "numpy":
        try:
            R = np.linalg.cholesky(K).T
        except np.linalg.LinAlgError:
            raise ValueError("Kernel matrix not positive!")
    elif method == "eigenpy":
        llt = eigenpy.LLT(K)
        R = llt.matrixL().T  # upper triangular matrix
    np.testing.assert_allclose(R.T @ R, K, atol=1e-10)
    return R, R_inv


class Problem:
    def __init__(
        self,
        lambd: float = 0.0,
        t: float = 1e-8,
        use_K: bool = False,
    ):
        self.lambd = lambd
        self.t = t  # epsilon / n_samples
        self.use_K = use_K

        # will be set later
        self.Phi: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.f_samples: np.ndarray | None = None
        self.samples: np.ndarray | None = None
        self.n_samples: int | None = None

        # currently not used anymore (this is for Section 7 in original paper)
        # but left for compatibility.
        self.nu = 0.0

    @property
    def epsilon(self):
        assert self.t is not None
        assert self.n_samples is not None
        return self.t * self.n_samples

    @staticmethod
    def create_random(N, k, epsilon=1e-8) -> "Problem":
        lambd = 1e-5

        Phi = np.random.rand(k, N)
        t = epsilon / N

        # create random psd B
        B_gt = np.random.rand(k, k)
        B_gt = (B_gt + B_gt.T) / 2  # make it symmetric
        B_gt = B_gt @ B_gt.T  # make it psd

        c_gt = 5

        f_samples = np.einsum("ik,ij,jk->k", Phi, B_gt, Phi) + c_gt
        samples = np.arange(N)[:, None]

        problem = Problem(
            lambd=lambd,
            t=t,
            use_K=True,
        )
        problem.register_fixed_samples(samples, f_samples=f_samples)
        problem.Phi = Phi
        problem.K = Phi.T @ Phi
        return problem

    def init_from_random(self, Phi: np.ndarray, f):
        K = Phi.T @ Phi
        k, N = Phi.shape
        self.samples = np.arange(N)[:, None]
        self.f_samples = f
        self.Phi = Phi
        self.K = K
        self.n_samples = N

    def get_M(self, alpha):
        # Recall that Phi = |phi_1 ... phi_N |
        assert self.Phi is not None
        return self.Phi @ (alpha[:, None] * self.Phi.T) + self.lambd * np.eye(
            self.Phi.shape[0]
        )

    def get_B(self, alpha, use_K=None):
        assert self.Phi is not None
        if use_K is None:
            use_K = self.use_K
        if use_K and self.lambd > 0:
            assert self.K is not None
            k = self.Phi.shape[0]

            to_invert = self.K + self.lambd * np.diag(1 / alpha)
            if np.linalg.cond(to_invert) > 1e10:
                warnings.warn(
                    "Ill-conditioning within B calculation, kernelized version"
                )
            B = (
                self.t
                / self.lambd
                * (np.eye(k) - self.Phi @ np.linalg.lstsq(to_invert, self.Phi.T)[0])
            )
        else:
            M = self.get_M(alpha)
            # print("minimum 3 eigenvalues of thing to invert", np.linalg.eigvalsh(M)[:3])
            B = self.t * np.linalg.pinv(M)
        return B

    def get_linear_system(self, alpha, c):
        assert self.Phi is not None
        assert self.f_samples is not None
        # create system of equations A @ vech(M) = b
        # created from the feasibility conditions, and the inverse condition.
        K, N = self.Phi.shape
        A = []
        b = []
        for i, f_i in enumerate(self.f_samples):
            phi_i = self.Phi[:, i]
            A.append(hvec(np.outer(phi_i, phi_i)))
            b.append(f_i - c)
        A_arr1 = np.array(A)
        b_arr1 = np.array(b).flatten()

        M_inv = self.get_M(alpha)
        D_n = duplication_matrix(K)
        A_arr2 = np.kron(M_inv.T, np.eye(K)) @ D_n
        b_arr2 = np.eye(K).flatten() * self.t

        A = np.vstack([A_arr1, A_arr2])
        b = np.hstack([b_arr1, b_arr2])
        return A, b

    def initialize_kernel(self, sigma, kernel, verbose=False, llt_method=LLT_METHOD):
        assert self.samples is not None

        # sanity check that sigma is roughly in the order of magnitude
        # of the distance between points
        D = np.linalg.norm(self.samples[None, ...] - self.samples[:, None, ...], axis=2)
        min_dist = np.min(D[np.triu_indices(n=D.shape[0], k=1)])
        if verbose:
            if sigma > min_dist * 2.0:
                print(
                    f"Warning, sigma is {sigma:.4e} which is more than 2 times the minimum distance {min_dist:.4e}"
                )
            elif sigma < min_dist / 2.0:
                print(
                    f"Warning, sigma is {sigma:.4e} which is less than half the minimum distance {min_dist:.4e}"
                )
            else:
                print(
                    f"Good choice: sigma {sigma:.4e} is close to minimum distance {min_dist:.4e}"
                )

        K = np.array(
            [
                [kernel_function(xi, xj, sigma, kernel) for xi in self.samples]
                for xj in self.samples
            ]
        )
        try:
            # Decompose K such that K = R'R, meaning that we have:
            # R' = |  phi_1' |   and R = | phi_1 ... phi_N |
            #      |    ...  |
            #      |  phi_N' |
            R, __ = decompose(K, method=llt_method)
            self.K = K
            self.Phi = R
            return True
        except (ValueError, AssertionError) as e:
            print(
                "Cholesky inaccurate -- is kernel matrix singular? Smallest 4 eigenvalues:"
            )
            print(np.linalg.eigvalsh(K)[:4])
            return False

    def initialize_kernel_from_Phi(self, Phi: np.ndarray):
        assert Phi.shape[1] == self.n_samples
        self.Phi = Phi
        self.K = self.Phi.T @ self.Phi

    def generate_new_samples(
        self,
        f: Callable[[np.ndarray], float],
        n_samples: int,
        center: np.ndarray = np.zeros(2),
        radius: np.ndarray | float = 1.0,
        sampling: str = "uniform",
        sampling_function: Callable[[], np.ndarray] | None = None,
    ):
        dim = len(center)
        if (radius is not None) and (
            isinstance(radius, float) or isinstance(radius, int)
        ):
            radius = np.array([radius] * dim)
        assert n_samples >= 1
        assert sampling in ["linspace", "uniform", "sobol"]

        # Generate samples
        if sampling_function is not None:
            assert isinstance(sampling_function, Callable)
            self.samples = np.array([sampling_function() for _ in range(n_samples)])
        else:
            self.samples = get_samples(center, radius, n_samples, sampling)
        self.f_samples = np.array([f(sj) for sj in self.samples]).flatten()
        self.n_samples = n_samples

    def register_fixed_samples(self, samples, f=None, f_samples=None):
        self.n_samples = samples.shape[0]
        # center = np.mean(samples, axis=0)
        # max_rad = np.max(samples - center[None, :])  # type: ignore
        # min_rad = np.min(samples - center[None, :])  # type: ignore
        # radius = np.max([max_rad, -min_rad])
        assert len(samples.shape) >= 2
        assert samples.shape[0] >= 1
        if f_samples is not None:
            assert f is None
            assert len(f_samples) == samples.shape[0]
            self.f_samples = np.array(f_samples).flatten()
        else:
            assert f is not None
            self.f_samples = np.array([f(sj) for sj in samples]).flatten()
        self.samples = samples

    def create_Phi_monomial(self, degree: int, samples: np.ndarray | None = None):
        """Create the feature matrix Phi using monomials up to given degree."""
        if samples is None:
            assert self.samples is not None
            samples = self.samples
        return get_monomial_vectors(degree, samples)
