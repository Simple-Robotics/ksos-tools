import cvxpy as cp
import numpy as np
from scipy.stats import qmc


def get_samples(center, radius, n_samples, sampling):
    dim = len(center)
    if isinstance(radius, float) or isinstance(radius, int):
        radius = np.array([radius] * dim)
    if sampling == "linspace":
        from math import ceil

        n_per_dim = ceil(n_samples ** (1 / dim))
        coords = np.meshgrid(
            *[
                np.linspace(center[i] - radius[i], center[i] + radius[i], n_per_dim)
                for i in range(dim)
            ]
        )
        samples = np.hstack([c.flatten()[:, None] for c in coords])
        samples = samples[:n_samples, :]
    elif sampling == "uniform":
        samples = np.array(
            [
                np.random.uniform(
                    center[i] - radius[i], center[i] + radius[i], n_samples
                )
                for i in range(dim)
            ]
        ).T
    elif sampling == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True)
        samples_unit = sampler.random(n_samples)
        samples = qmc.scale(
            samples_unit,
            l_bounds=center - radius,
            u_bounds=center + radius,
        )
    else:
        raise ValueError(f"Unknown sampling function: {sampling}")
    return samples


def duplication_matrix(n: int, scale: bool = True) -> np.ndarray:
    """
    Vectorized construction of the duplication matrix D_n
    of size (n^2, n(n+1)//2) satisfying vec(S) = D_n vech(S)
    for any symmetric n×n matrix S.

    Note that vech(S) includes scaling of off-diagonal elements by sqrt(2),
    so that we have <X, Y> = vech(X)^T vech(Y) for symmetric X, Y.
    """
    p = n * (n + 1) // 2
    # indices of lower triangle (including diagonal)
    lt_rows, lt_cols = np.tril_indices(n)
    # column index in vech for each lower–triangle pair
    col_idx = np.arange(p)

    # initialize with zeros
    D = np.zeros((n * n, p), dtype=int)

    # For each lower–triangle element, set two (or one) positions
    # Flattened positions in vec(S)
    flat_ij = lt_rows + n * lt_cols
    flat_ji = lt_cols + n * lt_rows

    D[flat_ij, col_idx] = 1
    D[flat_ji, col_idx] = 1  # includes diagonal, but that’s fine
    if scale:
        D = D / np.sqrt(D.sum(axis=0, keepdims=True))
    return D


def hvec(A, scale: bool = True):
    if scale:
        indices = np.tril_indices(A.shape[0])
        if isinstance(A, np.ndarray):
            vech = A[indices]
            vech[np.where(indices[0] != indices[1])] *= np.sqrt(2)
            return vech
        else:  # cp.Expression
            vech = []
            for r, c in zip(*indices):
                if r != c:
                    vech.append(A[r, c] * np.sqrt(2))
                else:
                    vech.append(A[r, c])
            return cp.hstack(vech)
    else:
        return A[np.tril_indices(A.shape[0])]
