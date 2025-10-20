"""Simple K-Means clustering implementation using NumPy."""

import numpy as np


def assign_centers(X: np.ndarray, centers: np.ndarray):
    """Assign each data point to the nearest cluster center."""
    sq_dists = np.sum((X[:, None, :] - centers[None]) ** 2, axis=-1)
    labels = np.argmin(sq_dists, axis=-1)
    return labels


def update_centers(labels: np.ndarray, centers: np.ndarray):
    """Update cluster centers based on current assignments."""
    return np.array([X[labels == k].mean(axis=0) for k in range(len(centers))])


def kmeans(
    X: np.ndarray, num_clusters: int, max_iters: int, tolerance: float = 1e-10
) -> np.ndarray:
    """Run K-Means clustering algorithm."""
    # Randomly initialize cluster centers
    centers = X[np.random.choice(a=np.arange(len(X)), replace=False, size=num_clusters)]
    labels = assign_centers(X, centers)

    # Iterate to refine centers and labels
    for k in range(max_iters):
        new_centers = update_centers(labels, centers)
        labels = assign_centers(X, new_centers)

        rel_change = np.sum((new_centers - centers) ** 2) / np.sum(centers**2)
        centers = new_centers
        if rel_change < tolerance:
            print(f"Converged after {k} iterations.")
            break

    return labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.concatenate(
        [
            np.random.randn(1000, 2),
            np.random.randn(1000, 2) * 0.5 + 2.0,
            np.random.randn(1000, 2) * 0.5 + 4.0,
        ],
        axis=0,
    )

    labels = kmeans(X, num_clusters=3, max_iters=100)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10")
    plt.show()
