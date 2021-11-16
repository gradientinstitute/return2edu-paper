import unittest
import numpy as np
from numpy.random import default_rng
from cov import covariance_matrix, cov_to_corr


class TestCovarianceMatrix(unittest.TestCase):
    def test_correlation_matrix_spectrum_preserved(self):
        rng = default_rng(seed=0)
        for _ in range(100):
            spectrum = rng.random(3)
            cov = covariance_matrix(spectrum)
            cov_evals = np.linalg.eigvals(cov)
            spectrum_sorted = np.sort(spectrum)
            cov_evals_sorted = np.sort(cov_evals)
            self.assertTrue(np.all(np.isclose(cov_evals_sorted, spectrum_sorted)))

    def test_correlation_matrix_symmetric(self):
        # TODO
        pass

    def test_cov_to_corr(self):
        rng = default_rng(seed=0)
        for _ in range(100):
            n = 100
            x = rng.standard_normal((n, 1))
            y = rng.normal(rng.random() * x)
            z = x + y
            data = np.hstack((x, y, z))
            cov = np.cov(data)
            cor = np.corrcoef(data)
            cor_from_cov = cov_to_corr(cov)
            self.assertTrue(np.all(np.isclose(cor, cor_from_cov)))

    def test_numerical_properties(self):
        # TODO
        pass


if __name__ == "__main__":
    unittest.main()
