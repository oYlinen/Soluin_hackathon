#!/usr/bin/env python3

import csv
from pathlib import Path
import time

import numpy as np
import viser

from Pointcloud_Highlight_test import (
    OBJECT_PALETTE,
    load_object_points,
)
from highlight_config import OBJECT_IDS_TO_HIGHLIGHT
from object_detection_config import OBJECT_POINTS_PATH
from test_object_detection import load_gaussians
from viser import BoxHandle, GaussianSplatHandle, SceneNodePointerEvent


PROJECT_DIR = Path(__file__).resolve().parent
PLY_PATH = PROJECT_DIR.parent / "Equipment Room.ply"
Y_AXIS_FLIP = np.array([1.0, -1.0, 1.0], dtype=np.float32)
HIGHLIGHT_SIGMA = 0.006
HIGHLIGHT_OPACITY = 0.15
SIMILARITIES_PATH = PROJECT_DIR / "similarities.csv"
EXCLUDED_LABEL = "not in targets"


def flip_gaussians_y(
    centers: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flipped_centers = centers.copy()
    flipped_centers[:, 1] *= -1
    flipped_covariances = (
        covariances
        * Y_AXIS_FLIP[np.newaxis, :, np.newaxis]
        * Y_AXIS_FLIP[np.newaxis, np.newaxis, :]
    )
    return flipped_centers, flipped_covariances


def load_object_labels(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Object similarity labels not found: {path}")

    labels: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"query_id", "label"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{path} is missing CSV columns: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            try:
                object_id = int(row["query_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path} contains an invalid query_id: "
                    f"{row['query_id']!r}"
                ) from error
            label = (row["label"] or "").strip()
            if not label:
                raise ValueError(
                    f"{path} contains an empty label for object {object_id}"
                )
            if object_id in labels:
                raise ValueError(
                    f"{path} contains duplicate labels for object {object_id}"
                )
            labels[object_id] = label

    return labels


def colors_for_objects(
    object_ids: np.ndarray,
    labels_by_object: dict[int, str],
    colors_by_label: dict[str, np.ndarray],
) -> np.ndarray:
    colors = np.empty((len(object_ids), 3), dtype=np.float32)
    for cluster_id in np.unique(object_ids):
        colors[object_ids == cluster_id] = colors_by_label[
            labels_by_object[int(cluster_id)]
        ]
    return colors


def select_configured_object_ids(
    cluster_ids: np.ndarray,
    configured_ids: list[int] | None,
) -> np.ndarray:
    if configured_ids is None:
        return cluster_ids

    requested_ids = np.asarray(configured_ids, dtype=np.int64)
    missing_ids = np.setdiff1d(requested_ids, cluster_ids)
    if len(missing_ids):
        raise ValueError(
            f"Configured object IDs are not present in the CSV: {missing_ids.tolist()}"
        )
    return np.intersect1d(cluster_ids, requested_ids)


def outer_eight_point_mask(
    object_ids: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    highlight_mask = np.zeros(len(object_ids), dtype=bool)
    for cluster_id in np.unique(object_ids):
        indices = np.flatnonzero(object_ids == cluster_id)
        points = object_points[indices]
        unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
        centroid = unique_points.mean(axis=0)
        offsets = unique_points - centroid
        octants = (
            (offsets[:, 0] >= 0).astype(np.int8)
            | ((offsets[:, 1] >= 0).astype(np.int8) << 1)
            | ((offsets[:, 2] >= 0).astype(np.int8) << 2)
        )
        distances_squared = np.einsum("ij,ij->i", offsets, offsets)

        for octant in range(8):
            octant_indices = np.flatnonzero(octants == octant)
            if len(octant_indices):
                farthest = octant_indices[np.argmax(distances_squared[octant_indices])]
                highlight_mask[indices[unique_indices[farthest]]] = True
    return highlight_mask


def main() -> None:
    if not PLY_PATH.is_file():
        raise FileNotFoundError(f"Gaussian PLY file not found: {PLY_PATH}")

    centers, covariances, colors, opacities = load_gaussians(PLY_PATH)
    object_ids, object_points = load_object_points(OBJECT_POINTS_PATH)
    cluster_ids = np.unique(object_ids)
    labels_from_csv = load_object_labels(SIMILARITIES_PATH)
    labels_by_object = {
        int(cluster_id): labels_from_csv.get(int(cluster_id), "Unlabeled")
        for cluster_id in cluster_ids
    }
    labels = sorted(
        label for label in set(labels_by_object.values()) if label != EXCLUDED_LABEL
    )
    colors_by_label = {
        label: OBJECT_PALETTE[index % len(OBJECT_PALETTE)].astype(np.float32)
        for index, label in enumerate(labels)
    }
    configured_cluster_ids = select_configured_object_ids(
        cluster_ids, OBJECT_IDS_TO_HIGHLIGHT
    )
    configured_cluster_ids = np.asarray(
        [
            cluster_id
            for cluster_id in configured_cluster_ids
            if labels_by_object[int(cluster_id)] != EXCLUDED_LABEL
        ],
        dtype=cluster_ids.dtype,
    )

    #outer_points_mask = outer_eight_point_mask(object_ids, object_points)
    highlighted_object_ids = object_ids#[outer_points_mask]
    highlighted_object_points = object_points#[outer_points_mask]

    rendered_centers, rendered_covariances = flip_gaussians_y(centers, covariances)
    rendered_object_points = highlighted_object_points.copy()
    rendered_object_points[:, 1] *= -1
    highlight_covariance = np.eye(3, dtype=np.float32) * HIGHLIGHT_SIGMA**2
    highlight_covariances = np.broadcast_to(
        highlight_covariance, (len(rendered_object_points), 3, 3)
    ).copy()
    highlight_opacities = np.full(
        (len(rendered_object_points), 1), HIGHLIGHT_OPACITY, dtype=np.float32
    )

    print(
        f"Loaded {len(centers):,} Gaussian splats and matched "
        f"{len(object_points):,} CSV points across {len(cluster_ids)} objects; "
        f"highlighting {len(configured_cluster_ids)} configured objects after "
        f"excluding label '{EXCLUDED_LABEL}'."
    )
    server = viser.ViserServer(host="127.0.0.1")
    splat_extents = 3.0 * np.sqrt(
        np.maximum(np.diagonal(rendered_covariances, axis1=1, axis2=2), 0.0)
    )
    bounds_min = np.min(rendered_centers - splat_extents, axis=0)
    bounds_max = np.max(rendered_centers + splat_extents, axis=0)
    cloud_center = (bounds_min + bounds_max) * 0.5
    initial_forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    server.initial_camera.position = cloud_center
    server.initial_camera.look_at = cloud_center + initial_forward
    server.initial_camera.up = (0.0, 1.0, 0.0)

    selector = server.gui.add_dropdown(
        "Highlight",
        [
            "All configured objects",
            *(f"Object {cluster_id}" for cluster_id in configured_cluster_ids),
        ],
        initial_value="All configured objects",
    )
    server.gui.add_markdown(
        "**Camera movement**\n\n"
        "- W / S: move forward / backward along the camera's facing direction\n"
        "- A / D: strafe left / right relative to the camera\n\n"
        "Click the viewer to focus it before using the keys. Selecting an object "
        "only changes its highlight; it does not move the camera. Click an "
        "object's bounding box to select it and show its information, or select "
        "it in the menu."
    )
    object_info = server.gui.add_markdown(
        "### Object information\n\nClick an object's box or select it in the "
        "Highlight menu to see its details."
    )

    server.scene.add_gaussian_splats(
        "/gaussians",
        rendered_centers,
        rendered_covariances,
        colors,
        opacities,
    )
    object_handles: dict[int, GaussianSplatHandle] = {}
    object_boxes: dict[int, BoxHandle] = {}
    for cluster_id in configured_cluster_ids:
        object_mask = highlighted_object_ids == cluster_id
        object_points = rendered_object_points[object_mask]
        object_handle = server.scene.add_gaussian_splats(
            f"/highlighted-objects/object-{cluster_id}",
            object_points,
            highlight_covariances[object_mask],
            colors_for_objects(
                highlighted_object_ids[object_mask],
                labels_by_object,
                colors_by_label,
            ),
            highlight_opacities[object_mask],
        )
        object_handles[int(cluster_id)] = object_handle

        object_min = object_points.min(axis=0)
        object_max = object_points.max(axis=0)
        object_extent = object_max - object_min
        padding = np.maximum(object_extent * 0.05, 0.01)
        box_dimensions = np.maximum(object_extent + 2.0 * padding, 0.02)
        box_handle = server.scene.add_box(
            f"/object-bounds/object-{cluster_id}",
            color=colors_by_label[labels_by_object[int(cluster_id)]],
            dimensions=box_dimensions,
            position=(object_min + object_max) * 0.5,
            wireframe=True,
            cast_shadow=False,
            receive_shadow=False,
            visible=True,
        )
        object_boxes[int(cluster_id)] = box_handle

        @box_handle.on_click
        def select_object_from_box(
            _event: SceneNodePointerEvent[BoxHandle],
            object_id: int = int(cluster_id),
        ) -> None:
            selector.value = f"Object {object_id}"

    @selector.on_update
    def update_highlight(_event: object) -> None:
        choice = selector.value
        selected_cluster_ids = (
            configured_cluster_ids
            if choice == "All configured objects"
            else np.array([int(choice.removeprefix("Object "))])
        )
        for object_id, handle in object_handles.items():
            handle.visible = object_id in selected_cluster_ids
        for object_id, handle in object_boxes.items():
            handle.visible = object_id in selected_cluster_ids
        if choice == "All configured objects":
            object_info.content = (
                "### Object information\n\nClick an object's box or select one "
                "object in the Highlight menu to see its details."
            )
            return

        object_id = int(choice.removeprefix("Object "))
        points = rendered_object_points[highlighted_object_ids == object_id]
        centroid = points.mean(axis=0)
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        object_info.content = (
            f"### Object {object_id}\n\n"
            f"- Label: **{labels_by_object[object_id]}**\n"
            f"- Highlight points: {len(points):,}\n"
            f"- Center (rendered X, Y, Z): "
            f"`({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})`\n"
            f"- Bounds min: "
            f"`({bounds_min[0]:.3f}, {bounds_min[1]:.3f}, {bounds_min[2]:.3f})`\n"
            f"- Bounds max: "
            f"`({bounds_max[0]:.3f}, {bounds_max[1]:.3f}, {bounds_max[2]:.3f})`"
        )

    @server.on_client_connect
    def initialize_client_camera(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0.0, 1.0, 0.0)
        client.camera.position = cloud_center
        client.camera.look_at = cloud_center + initial_forward

    print(f"Open http://localhost:{server.get_port()} to view the Gaussian splats.")
    print("Use the Highlight menu to show configured objects or one object at a time.")
    print("Click an object's bounding box to select it and display its information.")
    print("Camera movement: W/S forward/back, A/D strafe left/right.")
    print("The point overlay and Gaussian splats both use the mirrored Y axis.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
