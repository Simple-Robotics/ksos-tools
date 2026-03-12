import matplotlib
import matplotlib.colors
import matplotlib.pylab as plt
import numpy as np
import pytest
import os

from ksos_tools.examples.benchmarks import ackley, rosenbrock, schwefel
from ksos_tools.solvers import ksos

# Set matplotlib backend based on environment to avoid display issues
if os.environ.get("DISPLAY", "") == "":
    # Headless environment (e.g., CI)
    matplotlib.use("Agg")
else:
    # Local environment with GUI
    matplotlib.use("TkAgg")

DEFAULT_PARAMS = dict(
    sampling="linspace", return_all=False, return_B=False, verbose=False
)


def plot_solutions(center, radius, info, x_gt, f):
    # plot sample distributions vs. original cost
    n_plot = 100
    x = np.linspace(center[0] - radius, center[0] + radius, n_plot)
    y = np.linspace(center[1] - radius, center[1] + radius, n_plot)
    X = np.meshgrid(x, y)
    z = f(np.array(X))
    fig, ax = plt.subplots()
    ax.pcolorfast(x, y, z[:-1, :-1], alpha=0.5, norm=matplotlib.colors.LogNorm())
    ax.scatter(*x_gt, color="k", marker="x", s=50)
    for warm_it in range(len(info["samples"])):
        ax.scatter(*info["samples"][warm_it].T, color=f"C{warm_it}", marker="+")
        ax.scatter(*info["centers"][warm_it], color=f"C{warm_it}", marker="*", s=50)
    plt.show(block=False)


@pytest.mark.parametrize(
    "solver", ("MOSEK", "newton", "newton-features", "newton-kernel")
)
@pytest.mark.parametrize("kernel", ("Laplace", "Gauss"))
def test_ackley(solver, kernel, plot=False):
    f_here = lambda x: ackley(x)  # type: ignore # noqa: E731
    dim = 2
    x_gt = np.zeros(dim)

    center = np.zeros(dim) + 0.5
    radius = 32
    n_samples = 20

    # rule of thumb for sigma
    sigma = 2 * radius / (n_samples ** (1 / dim))
    warm_iterations = 5

    x_hat, info = ksos.solve(
        f_here,
        dim=dim,
        center=center,
        radius=radius,
        n_samples=n_samples,
        warm_iterations=warm_iterations,
        sigma=sigma,
        decay=0.5,
        solver=solver,
        kernel=kernel,
        **DEFAULT_PARAMS,  # type: ignore
    )
    assert x_hat is not None

    if plot:
        plot_solutions(center, radius, info, x_gt, f_here)

    np.testing.assert_allclose(x_hat, x_gt, atol=0.5)
    return


# Disabeling this test on purpose becuase it is too slow. It should however pass if the others
# are passing.
def notest_schwefel(solver, kernel, plot=False):
    def f_here(x):
        """
        We create a "barrier" because otherwise the solutions go outside of the
        500 x 500 box (the function becomes larger and larger as the radius is
        increased).
        """
        if np.ndim(x) == 1:
            if np.any(np.abs(x) > 500):
                return 1e5
            else:
                return schwefel(x)
        else:
            outside = np.any(np.abs(x) >= 500, axis=0)
            z = schwefel(x)
            z[outside] = np.max(z)
            return z

    dim = 2
    x_gt = np.full(dim, 420.9687)
    # adding +0.1 because otherwise minimum is on one of the points.
    center = np.zeros(dim)
    radius = 500
    n_samples = 50

    # rule of thumb for sigma
    sigma = 2 * radius / (n_samples ** (1 / dim))
    warm_iterations = 5

    x_hat, info = ksos.solve(
        f_here,
        dim=dim,
        center=center,
        radius=radius,
        n_samples=n_samples,
        warm_iterations=warm_iterations,
        decay=0.8,
        sigma=sigma,
        solver=solver,
        kernel=kernel,
        **DEFAULT_PARAMS,  # type: ignore
    )
    assert x_hat is not None

    if plot:
        plot_solutions(center, radius, info, x_gt, f_here)

    np.testing.assert_allclose(x_hat, x_gt, rtol=1e-1)
    return


@pytest.mark.parametrize(
    "solver", ("MOSEK", "newton", "newton-features", "newton-kernel")
)
@pytest.mark.parametrize("kernel", ("Laplace", "Gauss"))
def test_rosenbrock(solver, kernel, plot=False):
    a = 1.0
    b = 100
    f_here = lambda x: rosenbrock(x, a=a, b=b)  # type: ignore
    x_gt = np.array([a, a**2])
    dim = 2

    # adding +0.1 because otherwise minimum is on one of the points.
    center = np.ones(dim) + 0.1
    radius = 1
    n_samples = 20

    # rule of thumb for sigma
    sigma = 2 * radius / (n_samples ** (1 / dim))
    warm_iterations = 5

    x_hat, info = ksos.solve(
        f_here,
        dim=dim,
        center=center,
        radius=radius,
        n_samples=n_samples,
        warm_iterations=warm_iterations,
        decay=0.5,
        sigma=sigma,
        solver=solver,
        kernel=kernel,
        **DEFAULT_PARAMS,  # type: ignore
    )
    assert x_hat is not None

    if plot:
        plot_solutions(center, radius, info, x_gt, f_here)

    np.testing.assert_allclose(x_hat, x_gt, atol=0.5)
    return


if __name__ == "__main__":
    import itertools

    # test_ackley(solver="newton-new", kernel="Gauss", plot=True)
    # test_rosenbrock(solver="newton-new", kernel="Gauss", plot=True)
    # test_schwefel(solver="newton-new", kernel="Gauss", plot=True)
    plot = False
    for solver, kernel in itertools.product(
        ["MOSEK", "newton", "newton-kernel", "newton-features"], ["Gauss", "Laplace"]
    ):
        print(f"========== testing {solver} {kernel} =============")
        print(f"----------         Ackley            -------------")
        test_ackley(solver=solver, kernel=kernel, plot=plot)
        print(f"----------       Rosenbrock          -------------")
        test_rosenbrock(solver=solver, kernel=kernel, plot=plot)
        print(f"----------        Schwefel           -------------")
        notest_schwefel(solver=solver, kernel=kernel, plot=plot)
    print("all tests passed")
