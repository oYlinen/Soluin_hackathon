"""Shared scene constants and loaders (Gaussian PLY and detected-object CSV)."""

from pathlib import Path

import numpy as np
from plyfile import PlyData

PROJECT_DIR = Path(__file__).resolve().parent
PLY_PATH = PROJECT_DIR/ "Equipment Room.ply"
Y_AXIS_FLIP = np.array([1.0, -1.0, 1.0], dtype=np.float32)
SH_C0 = 0.28209479177387814
REQUIRED_FIELDS = {
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
}
OBJECT_PALETTE = np.array(
    [
        [1.00, 0.20, 0.12],
        [0.10, 0.85, 0.30],
        [0.10, 0.55, 1.00],
        [1.00, 0.75, 0.10],
        [0.85, 0.20, 1.00],
        [0.10, 0.90, 0.90],
        [1.00, 0.40, 0.70],
        [0.60, 1.00, 0.15],
    ],
    dtype=np.float64,
)


def quaternions_to_rotations(quaternions: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions in (w, x, y, z) order to (N, 3, 3) matrices."""
    w, x, y, z = quaternions.T
    rotations = np.empty((len(quaternions), 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * w)
    rotations[:, 0, 2] = 2 * (x * z + y * w)
    rotations[:, 1, 0] = 2 * (x * y + z * w)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * w)
    rotations[:, 2, 0] = 2 * (x * z - y * w)
    rotations[:, 2, 1] = 2 * (y * z + x * w)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def load_gaussians(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load centers, covariances, RGB colors and opacities from a Gaussian PLY."""
    vertices = PlyData.read(path, mmap=True)["vertex"].data
    missing = REQUIRED_FIELDS.difference(vertices.dtype.names or ())
    if missing:
        raise ValueError(f"{path} is missing Gaussian fields: {', '.join(sorted(missing))}")

    def columns(*names: str) -> np.ndarray:
        return np.column_stack([vertices[name] for name in names]).astype(np.float32)

    centers = columns("x", "y", "z")
    log_scales = columns("scale_0", "scale_1", "scale_2")
    quaternions = columns("rot_0", "rot_1", "rot_2", "rot_3")

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{path} contains a Gaussian with a zero-length rotation quaternion")
    rotations = quaternions_to_rotations(quaternions / norms)

    variances = np.exp(2 * log_scales)
    covariances = (rotations * variances[:, np.newaxis, :]) @ np.swapaxes(rotations, 1, 2)
    colors = np.clip(0.5 + SH_C0 * columns("f_dc_0", "f_dc_1", "f_dc_2"), 0.0, 1.0)
    opacities = (1.0 / (1.0 + np.exp(-columns("opacity"))))

    if not all(np.isfinite(a).all() for a in (centers, covariances, colors, opacities)):
        raise ValueError(f"{path} contains non-finite Gaussian values")
    return centers, covariances, colors, opacities


def load_object_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (cluster_ids, points) from the detected-object CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Detected-object CSV not found: {path}. Run test_object_detection.py "
            "and click Detect objects first."
        )

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0 or data.dtype.names is None:
        raise ValueError(f"{path} contains no detected object points")
    data = np.atleast_1d(data)

    missing = {"cluster_id", "x", "y", "z"}.difference(data.dtype.names)
    if missing:
        raise ValueError(f"{path} is missing CSV columns: {', '.join(sorted(missing))}")

    cluster_ids = np.asarray(data["cluster_id"], dtype=np.float64)
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    if not np.isfinite(cluster_ids).all() or not np.isfinite(points).all():
        raise ValueError(f"{path} contains non-finite cluster IDs or coordinates")
    if not np.equal(cluster_ids, np.floor(cluster_ids)).all():
        raise ValueError(f"{path} contains non-integer cluster IDs")
    return cluster_ids.astype(np.int32), points
