import numpy as np

from ksos_tools.utils import duplication_matrix, hvec


def test_duplication_matrix():
    for n in range(1, 6):

        for scale in [True, False]:
            D = duplication_matrix(n, scale=scale)
            p = n * (n + 1) // 2
            assert D.shape == (n * n, p)

            # test that it works as intended
            S = np.random.rand(n, n)
            S = (S + S.T) / 2  # make symmetric

            vech_S = hvec(S, scale=scale)
            vec_S = S.flatten()
            np.testing.assert_allclose(D @ vech_S, vec_S)
    return


def test_hvec():
    for n in range(1, 6):
        A = np.random.rand(n, n)
        A = (A + A.T) / 2  # make symmetric

        B = np.random.rand(n, n)
        B = (B + B.T) / 2  # make symmetric

        vech_a = hvec(A, scale=True)
        vech_b = hvec(B, scale=True)

        prod1 = np.trace(A @ B)
        prod2 = vech_a @ vech_b
        np.testing.assert_allclose(prod1, prod2)  # type: ignore
    return


if __name__ == "__main__":
    test_hvec()
    test_duplication_matrix()
    print("all tests passed")
