"""Subsampling by minimizing Maximum Mean Discrepancy (MMD).

This script compares Numpy and CuPy implementations of a greedy MMD-based subsampling algorithm.
"""

import cupy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def _pairwise_distances_gemm(X: cp.ndarray) -> cp.ndarray:
    # X: [n, d], float32
    # returns D: [n, n] with Euclidean distances
    sq_norms = cp.sum(X * X, axis=1, keepdims=True)  # [n,1]
    G = X @ X.T  # [n,n] GEMM (fast)
    dist2 = cp.maximum(sq_norms + sq_norms.T - 2.0 * G, 0)  # numerical clamp
    return cp.sqrt(dist2)


def _pairwise_distances_gemm(X: cp.ndarray) -> cp.ndarray:
    # X: [n, d], float32
    # returns D: [n, n] Euclidean distances
    sq = cp.sum(X * X, axis=1, keepdims=True)  # [n,1]
    G = X @ X.T  # [n,n] cuBLAS GEMM
    d2 = cp.maximum(sq + sq.T - 2.0 * G, 0.0)  # clamp
    return cp.sqrt(d2)


def mmd_subsample_cupy(
    X: NDArray[np.float32] | cp.ndarray,
    size: int,
) -> NDArray[np.int64]:
    # Accept NumPy or CuPy X; convert once.
    X_cp = X if isinstance(X, cp.ndarray) else cp.asarray(X, dtype=cp.float32)
    n = X_cp.shape[0]
    assert size <= n

    # Precompute norms, distances (1x), and k0_mean via means (no Gram)
    norms = cp.linalg.norm(X_cp, axis=1)  # [n]
    dist = _pairwise_distances_gemm(X_cp)  # [n,n]
    mean_norm = cp.mean(norms)  # scalar
    mean_dist = cp.mean(dist, axis=1)  # [n]
    k0_mean = norms + mean_norm - mean_dist  # [n]

    # Greedy state
    idx = cp.empty(size, dtype=cp.int64)
    # k0[:,0] = 2 * norms ; we never need to store all k0 columns
    base = 2.0 * norms  # [n]
    s = cp.zeros_like(norms)  # [n] running sum of k0[:,1..t]
    alpha = -2.0 * k0_mean  # [n]

    # t = 0
    obj0 = base + alpha  # since -(t+1)= -1 times 2*k0_mean -> alpha
    idx[0] = cp.argmin(obj0)

    # Loop
    for t in range(1, size):
        j_prev = idx[t - 1]
        # k0[:, t] = -dist[:, j_prev] + norms[j_prev] + norms
        # Accumulate into s in-place (no storing all columns)
        s += -dist[:, j_prev] + norms[j_prev] + norms
        # obj_t = base + 2*s + (t+1)*alpha
        obj_t = base + (2.0 * s) + (t + 1) * alpha
        idx[t] = cp.argmin(obj_t)

    return cp.asnumpy(idx)


def mmd_subsample_fn(
    X: NDArray[np.float64],
    size: int,
) -> NDArray[np.int64]:
    """Selects samples in the input table by greedily minimizing the maximum mean discrepancy (MMD)."""
    n = X.shape[0]
    assert size <= n

    # Precompute norms and distance matrix
    norms = np.linalg.norm(X, axis=1)
    dist_matrix = cdist(X, X)
    gram_matrix = norms[:, None] + norms[None, :] - dist_matrix
    k0_mean = np.mean(gram_matrix, axis=1)

    idx = np.zeros(size, dtype=np.int64)
    k0 = np.zeros((n, size))
    k0[:, 0] = 2.0 * norms

    idx[0] = np.argmin(k0[:, 0] - 2.0 * k0_mean)
    for i in range(1, size):
        x_ = X[idx[i - 1]]
        dist = np.linalg.norm(X - x_, axis=1)
        k0[:, i] = -dist + norms[idx[i - 1]] + norms

        idx[i] = np.argmin(
            k0[:, 0]
            + 2.0 * np.sum(k0[:, 1 : (i + 1)], axis=1)
            - 2.0 * (i + 1) * k0_mean
        )
    return idx


def main() -> None:
    """Compares the performance of Numpy and CuPy implementations."""
    import time

    n = 10000
    d = 100
    size = 100

    X = np.random.randn(n, d).astype(np.float32)

    mean_time = 0.0
    for _ in range(10):
        start = time.time()
        _ = mmd_subsample_fn(X, size)
        end = time.time()
        mean_time += end - start
    mean_time /= 10.0
    print(f"Numpy (not memory safe), mean over 10 runs: {mean_time:.4f} seconds")

    X = cp.asarray(X, dtype=cp.float32)
    _ = mmd_subsample_cupy(X, size)
    cp.cuda.Stream.null.synchronize()

    mean_time = 0.0
    for _ in range(10):
        start = time.time()
        _ = mmd_subsample_cupy(X, size)
        cp.cuda.Stream.null.synchronize()  # ensure GPU work is done
        end = time.time()
        mean_time += end - start
    mean_time /= 10.0
    print(f"CuPy (memory safe), mean over 10 runs: {mean_time:.4f} seconds")


if __name__ == "__main__":
    main()
