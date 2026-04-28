import numpy as np
from scipy.spatial import cKDTree

def best_fit_transform(A: np.ndarray, B: np.ndarray):
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.shape[1] != 3:
        raise ValueError("Points must be 3D")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, R, t

def apply_transform(points: np.ndarray, T: np.ndarray):
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    hom = np.hstack([points, ones])
    transformed = (T @ hom.T).T
    return transformed[:, :3]

def icp(source: np.ndarray, target: np.ndarray, init_transform: np.ndarray | None = None,
        max_iterations: int = 50, tolerance: float = 1e-6, max_correspondence_distance: float | None = None,):

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source must be shape (N, 3)")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target must be shape (M, 3)")

    src = source.copy()
    dst = target.copy()

    if init_transform is None:
        T_total = np.eye(4)
    else:
        if init_transform.shape != (4, 4):
            raise ValueError("init_transform must be shape (4, 4)")
        T_total = init_transform.copy()
        src = apply_transform(src, T_total)

    tree = cKDTree(dst)
    prev_error = np.inf
    history = []

    for _ in range(max_iterations):
        distances, indices = tree.query(src, k=1)

        if max_correspondence_distance is not None:
            mask = distances <= max_correspondence_distance
            if mask.sum() < 3:
                raise RuntimeError("Too few correspondences after distance filtering")
            src_corr = src[mask]
            dst_corr = dst[indices[mask]]
            mean_error = distances[mask].mean()
        else:
            src_corr = src
            dst_corr = dst[indices]
            mean_error = distances.mean()
        T_step, _, _ = best_fit_transform(src_corr, dst_corr)

        # Update current source estimate
        src = apply_transform(src, T_step)

        # Accumulate transform
        T_total = T_step @ T_total

        history.append(float(mean_error))

        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return T_total, src, float(prev_error), history

if __name__ == "__main__":
    np.random.seed(42)

    # Create synthetic target cloud
    target = np.random.uniform(-1.0, 1.0, size=(2000, 3))

    # Ground-truth transform
    angle = np.deg2rad(12.0)
    R_gt = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    t_gt = np.array([0.35, -0.20, 0.10])

    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3] = t_gt

    # Build source by transforming target, then add small noise
    source = apply_transform(target, np.linalg.inv(T_gt))
    source += 0.003 * np.random.randn(*source.shape)

    T_est, source_aligned, err, hist = icp(
        source,
        target,
        max_iterations=60,
        tolerance=1e-8,
        max_correspondence_distance=0.25,
    )

    print("Ground truth transform:\n", T_gt)
    print("\nEstimated transform:\n", T_est)
    print("\nFinal mean error:", err)
