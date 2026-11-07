#!/usr/bin/env python3

import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import viser
from plyfile import PlyData

from object_detection_config import DBSCAN_EPS, DBSCAN_MIN_POINTS, OBJECT_POINTS_PATH

PLY_PATH = Path(__file__).resolve().parents[1] / "Equipment Room.ply"
SH_C0 = 0.28209479177387814


def save_object_points(
    centers: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> int:
    cluster_ids = np.unique(labels[labels >= 0])
    if len(cluster_ids) == 0:
        print("No object clusters found; no point data was saved.")
        return 0

    point_indices = np.flatnonzero(labels >= 0)
    points_with_ids = np.column_stack(
        (labels[point_indices], centers[point_indices])
    )
    np.savetxt(
        path,
        points_with_ids,
        delimiter=",",
        header="cluster_id,x,y,z",
        comments="",
        fmt=["%d", "%.9g", "%.9g", "%.9g"],
    )
    print(f"Saved points for {len(cluster_ids)} objects to {path}")
    return len(cluster_ids)


def load_gaussians(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = PlyData.read(path, mmap=True)["vertex"].data
    fields = vertices.dtype.names
    required_fields = {
        "x", "y", "z",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    }
    missing_fields = required_fields.difference(fields or ())
    if missing_fields:
        raise ValueError(f"{path} is missing Gaussian fields: {', '.join(sorted(missing_fields))}")

    centers = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    log_scales = np.column_stack(
        (vertices["scale_0"], vertices["scale_1"], vertices["scale_2"])
    ).astype(np.float32)
    quaternions = np.column_stack(
        (vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"])
    ).astype(np.float32)

    quaternion_norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(quaternion_norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{path} contains a Gaussian with a zero-length rotation quaternion")
    quaternions /= quaternion_norms

    w, x, y, z = quaternions.T
    rotations = np.empty((len(vertices), 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * w)
    rotations[:, 0, 2] = 2 * (x * z + y * w)
    rotations[:, 1, 0] = 2 * (x * y + z * w)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * w)
    rotations[:, 2, 0] = 2 * (x * z - y * w)
    rotations[:, 2, 1] = 2 * (y * z + x * w)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)

    variances = np.exp(2 * log_scales)
    covariances = (rotations * variances[:, np.newaxis, :]) @ np.swapaxes(rotations, 1, 2)
    colors = np.column_stack(
        (vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"])
    ).astype(np.float32)
    colors = np.clip(0.5 + SH_C0 * colors, 0.0, 1.0)
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-opacity_logits)))[:, np.newaxis]

    if not all(np.isfinite(array).all() for array in (centers, covariances, colors, opacities)):
        raise ValueError(f"{path} contains non-finite Gaussian values")
    return centers, covariances, colors, opacities


def focus_camera(
    camera: viser.CameraHandle,
    centers: np.ndarray,
    covariances: np.ndarray,
    distance_scale: float = 1.15,
) -> None:
    extents = 3.0 * np.sqrt(np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 0.0))
    bounds_min = np.min(centers - extents, axis=0)
    bounds_max = np.max(centers + extents, axis=0)
    target = (bounds_min + bounds_max) * 0.5
    radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 0.05)

    vertical_fov = camera.fov
    aspect = max(camera.aspect, 1e-3)
    horizontal_fov = 2.0 * np.arctan(np.tan(vertical_fov * 0.5) * aspect)
    limiting_fov = min(vertical_fov, horizontal_fov)
    distance = radius / np.sin(limiting_fov * 0.5) * distance_scale
    direction = np.array([1.0, -1.0, 0.75])
    direction /= np.linalg.norm(direction)

    camera.up_direction = (0.0, 0.0, 1.0)
    camera.position = target + direction * distance
    camera.look_at = target
    camera.min_orbit_distance = max(radius * 0.01, 0.01)
    camera.max_orbit_distance = max(distance * 20.0, 10.0)


def main() -> None:
    if not PLY_PATH.is_file():
        raise FileNotFoundError(f"Gaussian PLY file not found: {PLY_PATH}")

    centers, covariances, colors, opacities = load_gaussians(PLY_PATH)
    print(f"Loaded {len(centers):,} Gaussian splats from {PLY_PATH}")
    server = viser.ViserServer(host="127.0.0.1")
    selector = server.gui.add_dropdown(
        "Show", ["All Gaussians"], initial_value="All Gaussians"
    )
    detect_button = server.gui.add_button("Detect objects")
    clustering_lock = threading.Lock()
    labels: np.ndarray | None = None
    visible_indices = np.arange(len(centers))
    selected_splats = server.scene.add_gaussian_splats(
        "/gaussians",
        centers,
        covariances,
        colors,
        opacities,
    )

    @selector.on_update
    def update_selection(_event: object) -> None:
        nonlocal selected_splats, visible_indices
        choice = selector.value
        if choice == "All Gaussians":
            visible_indices = np.arange(len(centers))
        else:
            if labels is None:
                raise RuntimeError("Run object detection before selecting a cluster")
            cluster_id = int(choice.removeprefix("Object "))
            visible_indices = np.flatnonzero(labels == cluster_id)
        selected_splats.remove()
        selected_splats = server.scene.add_gaussian_splats(
            "/gaussians",
            centers[visible_indices],
            covariances[visible_indices],
            colors[visible_indices],
            opacities[visible_indices],
        )
        for client in server.get_clients().values():
            focus_camera(
                client.camera,
                centers[visible_indices],
                covariances[visible_indices],
            )

    @detect_button.on_click
    def detect_objects(_event: object) -> None:
        nonlocal labels
        with clustering_lock:
            if labels is not None:
                return
            print("Finding object clusters...")
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(centers)
            labels = np.asarray(
                point_cloud.cluster_dbscan(
                    eps=DBSCAN_EPS,
                    min_points=DBSCAN_MIN_POINTS,
                    print_progress=True,
                )
            )
            save_object_points(centers, labels, OBJECT_POINTS_PATH)
            cluster_ids = np.unique(labels[labels >= 0])
            selector.options = [
                "All Gaussians",
                *(f"Object {cluster_id}" for cluster_id in cluster_ids),
            ]
            print(f"Found: {len(cluster_ids)} objects")

    @server.on_client_connect
    def focus_new_client(client: viser.ClientHandle) -> None:
        focus_camera(
            client.camera,
            centers[visible_indices],
            covariances[visible_indices],
        )

    print(f"Open http://localhost:{server.get_port()} to view the Gaussian splats.")
    print("The full scene is shown immediately. Use Detect objects to enable cluster selection.")
    print("Press Ctrl+C to exit.")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
