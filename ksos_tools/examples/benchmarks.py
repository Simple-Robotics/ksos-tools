import numpy as np


def rosenbrock(X, a=1, b=100):
    x, y = X
    return (a - x) ** 2 + b * (y - x**2) ** 2


def schwefel(X):
    x, y = X
    return (
        418.9829 * 2 - x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))
    )


def ackley(X):
    x, y = X
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )


if __name__ == "__main__":
    import matplotlib.pylab as plt

    grid_coords = np.linspace(-1, 1, 1000)
    X = np.meshgrid(grid_coords, grid_coords)

    z = rosenbrock(X)
    fig, ax = plt.subplots()
    ax.pcolormesh(X[0], X[1], z)

    z = schwefel(X)
    fig, ax = plt.subplots()
    ax.pcolormesh(X[0], X[1], z)

    z = ackley(X)
    fig, ax = plt.subplots()
    ax.pcolormesh(X[0], X[1], z)
    plt.show()
