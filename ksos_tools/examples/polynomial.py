import matplotlib.pylab as plt
import numpy as np


class Polynomial(object):
    """Polynomial with global min at -0.5, local min at 0.5, and local max at 0.25"""

    def __init__(self):
        self.x_opt = np.array([-0.5])

    def cost(self, x):
        return (
            1 / 4 * (4 * x) ** 4
            - 1 / 3 * (4 * x) ** 3
            - 2 * (4 * x) ** 2
            + 4 * (4 * x)
            + 10
        )

    def get_valid_samples(self, n_samples):
        return np.linspace(-1, 1, n_samples)[:, None]
        return np.random.uniform(-1, 1, size=(n_samples, 1))

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        xs = np.arange(-1, 1, step=0.05)
        line = ax.plot(xs, self.cost(xs), **kwargs)
        ax.grid()
        return line[0]

    def __repr__(self):
        return f"Poly4"


if __name__ == "__main__":
    poly = Polynomial()
    poly.plot()
    plt.show(block=False)
    print("done")
