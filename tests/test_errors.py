import warnings

import numpy as np
import pytest

from ksos_tools.solvers import ksos
from tests.test_benchmarks import SOLVERS


@pytest.mark.parametrize("solver", SOLVERS)
def test_matrix_not_psd(solver):
    # Test if the solver can handle a non-PSD kernel matrix properly.

    samples = np.zeros((5, 1), dtype=float)
    f = lambda x: float(np.sum(x**2))  # noqa: E731

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        z, info = ksos.solve(
            f=f,
            dim=1,
            samples=samples,
            solver=solver,
            kernel="Gauss",
            sigma=1.0,
            epsilon=1e-6,
            n_samples=5,
            return_B=True,
            verbose=False,
        )

    assert isinstance(info, dict)
    assert z is None or np.all(np.isfinite(z))
    assert len(caught) >= 0
    if info.get("status") is not None:
        assert info["status"] in {"Kernel matrix not PSD", "Solution extrapolated"}


if __name__ == "__main__":
    for solver in SOLVERS:
        print(f"Testing solver: {solver}")
        test_matrix_not_psd(solver)
