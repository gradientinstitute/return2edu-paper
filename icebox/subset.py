import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # make redundant data with lots of columns
    n = 10000  # data
    d = 1000  # features

    r = 20  # latent dimensions
    w = np.random.randn(r, d)
    z = np.random.randn(n, r)
    noise = np.random.rand(1, d)  # per-column noise level
    data = z @ w + noise * np.random.randn(n, d)


​
    # Truncate if signal variance gets below 0.1
    ind, var = select_subset(data, tol=0.1)
​
    plt.plot(var)
    plt.title("Residual signal vs number of features")
    plt.show()
​
    keep_dims = 20
    data_short = data[:, ind[:keep_dims]]
    # TADA!
​
​


def select_subset(data, tol=1e-15):
    """Order the indicators to (greedily) maximise information.
    Parameters
    ----------
    data: n * d numpy array
    Returns:
    selection: list
        indices of data columns in order of information
        truncates at actual rank of dataset (no redundancy)
    variance: list
        total variance at each truncation
    """
    n, d = data.shape
    print("Computing covariance matrix...")
    cov = np.cov(data.T)
    select = np.arange(d)
    variance = np.zeros(d)


​
    print("Processing rows...")
    for i in tqdm(range(d - 1)):
        # total variance reduction
        # (there are some hand-calculations behind this,
        #  based on the update lines below)
        value = np.dot(cov ** 2, np.maximum(np.diag(cov), tol))
​
        # Select feature to maximise reduction
        target = np.argmax(value)
        variance[i] = value[target]
        if variance[i] < tol:
            break
        # swap to the front
        swap1 = [i, target]
        swap2 = [target, i]
        select[swap1] = select[swap2]
        cov[swap1] = cov[swap2]
        cov[:, swap1] = cov[:, swap2]
​
        # Update covariance of remaining rows conditioned on this one
        u = cov[i:, i] / np.sqrt(cov[i, i])
        cov[i:, i:] -= np.outer(u, u)
    i += 1  # Include the last column
    return select[:i], variance[:i]
