#!/usr/bin/env python3
"""Viser Gaussian-splat viewer with DBSCAN object detection (writes the object CSV)."""

import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import viser

from detection_config import DBSCAN_EPS, DBSCAN_MIN_POINTS, OBJECT_POINTS_PATH
from scene_data import PLY_PATH, load_gaussians

ALL_GAUSSIANS = "All Gaussians"


def save_object_points(centers: np.ndarray, labels: np.ndarray, path: Path) -> int:
    """Write clustered points (noise excluded) to CSV; return the cluster count."""
    cluster_ids = np.unique(labels[labels >= 0])
    if len(cluster_ids) == 0:
        print("No object clusters found; no point data was saved.")
        return 0

    indices = np.flatnonzero(labels >= 0)
    np.savetxt(
        path,
        np.column_stack((labels[indices], centers[indices])),
        delimiter=",",
        header="cluster_id,x,y,z",
        comments="",
        fmt=["%d", "%.9g", "%.9g", "%.9g"],
    )
    print(f"Saved points for {len(cluster_ids)} objects to {path}")
    return len(cluster_ids)


def cluster_gaussians(centers: np.ndarray) -> np.ndarray:
    """DBSCAN-cluster Gaussian centers; returns per-point labels (-1 = noise)."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(centers)
    return np.asarray(
        point_cloud.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=True,
        )
    )


def detect_and_save() -> int:
    """Headless detection (no viewer, no button click); returns the object count."""
    if not PLY_PATH.is_file():
        raise FileNotFoundError(f"Gaussian PLY file not found: {PLY_PATH}")
    centers, *_ = load_gaussians(PLY_PATH)
    print(f"Loaded {len(centers):,} Gaussian splats; finding object clusters...")
    labels = cluster_gaussians(centers)
    return save_object_points(centers, labels, OBJECT_POINTS_PATH)


def focus_camera(
    camera: viser.CameraHandle,
    centers: np.ndarray,
    covariances: np.ndarray,
    distance_scale: float = 1.15,
) -> None:
    """Frame the given Gaussians in the camera's view."""
    extents = 3.0 * np.sqrt(np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 0.0))
    bounds_min = (centers - extents).min(axis=0)
    bounds_max = (centers + extents).max(axis=0)
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
    selector = server.gui.add_dropdown("Show", [ALL_GAUSSIANS], initial_value=ALL_GAUSSIANS)
    detect_button = server.gui.add_button("Detect objects")

    clustering_lock = threading.Lock()
    labels: np.ndarray | None = None
    visible = np.arange(len(centers))

    def add_splats(indices: np.ndarray) -> viser.GaussianSplatHandle:
        return server.scene.add_gaussian_splats(
            "/gaussians",
            centers[indices],
            covariances[indices],
            colors[indices],
            opacities[indices],
        )

    def focus_clients() -> None:
        for client in server.get_clients().values():
            focus_camera(client.camera, centers[visible], covariances[visible])

    splats = add_splats(visible)

    @selector.on_update
    def _update_selection(_event: object) -> None:
        nonlocal splats, visible
        choice = selector.value
        if choice == ALL_GAUSSIANS:
            visible = np.arange(len(centers))
        else:
            if labels is None:
                raise RuntimeError("Run object detection before selecting a cluster")
            visible = np.flatnonzero(labels == int(choice.removeprefix("Object ")))
        splats.remove()
        splats = add_splats(visible)
        focus_clients()

    @detect_button.on_click
    def _detect_objects(_event: object) -> None:
        nonlocal labels
        with clustering_lock:
            if labels is not None:
                return
            print("Finding object clusters...")
            labels = cluster_gaussians(centers)
            save_object_points(centers, labels, OBJECT_POINTS_PATH)
            cluster_ids = np.unique(labels[labels >= 0])
            selector.options = [ALL_GAUSSIANS, *(f"Object {i}" for i in cluster_ids)]
            print(f"Found: {len(cluster_ids)} objects")

    @server.on_client_connect
    def _focus_new_client(client: viser.ClientHandle) -> None:
        focus_camera(client.camera, centers[visible], covariances[visible])

    print(f"Open http://localhost:{server.get_port()} to view the Gaussian splats.")
    print("The full scene is shown immediately. Use Detect objects to enable cluster selection.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
