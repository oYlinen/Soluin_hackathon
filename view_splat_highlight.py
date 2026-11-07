#!/usr/bin/env python3
"""Viser viewer: Gaussian splats with labeled, highlighted detected objects."""

import csv
import re
import time
from pathlib import Path

import numpy as np
import viser
from viser import BoxHandle, GaussianSplatHandle, SceneNodePointerEvent

from classification_config import SIMILARITIES_PATH
from detection_config import OBJECT_POINTS_PATH
from highlight_config import OBJECT_IDS_TO_HIGHLIGHT
from scene_data import (
    OBJECT_PALETTE,
    PLY_PATH,
    PROJECT_DIR,
    Y_AXIS_FLIP,
    load_gaussians,
    load_object_points,
)

ASSET_STATES_PATH = PROJECT_DIR / "current_state_with_asset_class.csv"
ASSET_FIELDS = (
    ("criticality", "Criticality"),
    ("condition_state", "Condition"),
    ("issue_code", "Issue code"),
    ("recommended_next_check", "Recommended next check"),
)

HIGHLIGHT_SIGMA = 0.006
HIGHLIGHT_OPACITY = 0.15
EXCLUDED_LABEL = "not in targets"
UNLABELED = "Unlabeled"
ALL_OBJECTS = "All configured objects"

ALL_OBJECTS_INFO = (
    "### Object information\n\nClick an object's box or select one "
    "object in the Highlight menu to see its details."
)
INSTRUCTIONS = (
    "**Camera movement**\n\n"
    "- W / S: move forward / backward along the camera's facing direction\n"
    "- A / D: strafe left / right relative to the camera\n\n"
    "Click the viewer to focus it before using the keys. Selecting an object "
    "only changes its highlight; it does not move the camera. Click an "
    "object's bounding box to select it and show its information, or select "
    "it in the menu."
)


# --------------------------------------------------------------------------
# Data loading / preparation
# --------------------------------------------------------------------------
def flip_gaussians_y(
    centers: np.ndarray, covariances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror Gaussian centers and covariances across the Y axis."""
    flipped_covariances = covariances * np.outer(Y_AXIS_FLIP, Y_AXIS_FLIP)
    return centers * Y_AXIS_FLIP, flipped_covariances


def load_object_labels(path: Path) -> dict[int, str]:
    """Read {query_id: label} from the similarities CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Object similarity labels not found: {path}")

    labels: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {"query_id", "label"}.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing CSV columns: {', '.join(sorted(missing))}")

        for row in reader:
            try:
                object_id = int(row["query_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path} contains an invalid query_id: {row['query_id']!r}"
                ) from error
            label = (row["label"] or "").strip()
            if not label:
                raise ValueError(f"{path} contains an empty label for object {object_id}")
            if object_id in labels:
                raise ValueError(f"{path} contains duplicate labels for object {object_id}")
            labels[object_id] = label
    return labels


def load_asset_states(path: Path) -> dict[str, list[dict[str, str]]]:
    """Read asset rows from the state CSV, grouped by lowercase asset_class."""
    if not path.is_file():
        raise FileNotFoundError(f"Asset state CSV not found: {path}")

    states: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"asset_class", *(field for field, _ in ASSET_FIELDS)}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            states.setdefault(row["asset_class"].strip().lower(), []).append(row)
    return states


def asset_class_for_label(label: str) -> str:
    """'Target Valve_handle' / 'Valve_handle' -> 'valve_handle'."""
    return re.sub(r"^target\s+", "", label.strip(), flags=re.IGNORECASE).lower()


def assign_asset_states(
    object_ids: list[int],
    labels_by_object: dict[int, str],
    states_by_class: dict[str, list[dict[str, str]]],
) -> dict[int, dict[str, str]]:
    """Give each object one state row from its label's asset class.

    Objects of the same class (in ID order) take that class's rows in turn,
    wrapping around when there are more objects than rows. Objects whose class
    has no rows in the CSV get no entry.
    """
    seen: dict[str, int] = {}
    assigned: dict[int, dict[str, str]] = {}
    for object_id in sorted(object_ids):
        asset_class = asset_class_for_label(labels_by_object[object_id])
        rows = states_by_class.get(asset_class)
        if not rows:
            continue
        index = seen.get(asset_class, 0)
        assigned[object_id] = rows[index % len(rows)]
        seen[asset_class] = index + 1
    return assigned


def select_object_ids(
    cluster_ids: list[int],
    configured_ids: list[int] | None,
    labels_by_object: dict[int, str],
) -> list[int]:
    """Objects to display: configured (or all) IDs, minus excluded labels."""
    if configured_ids is None:
        candidates = cluster_ids
    else:
        missing = sorted(set(configured_ids).difference(cluster_ids))
        if missing:
            raise ValueError(f"Configured object IDs are not present in the CSV: {missing}")
        candidates = sorted(set(configured_ids))
    return [i for i in candidates if labels_by_object[i] != EXCLUDED_LABEL]


def splat_bounds(centers: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of the splats, padded by 3 sigma."""
    extents = 3.0 * np.sqrt(np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 0.0))
    return (centers - extents).min(axis=0), (centers + extents).max(axis=0)


def describe_object(
    object_id: int,
    label: str,
    points: np.ndarray,
    asset: dict[str, str] | None,
) -> str:
    def fmt(v: np.ndarray) -> str:
        return f"`({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})`"

    if asset is None:
        asset_section = (
            f"\nNo entry for asset class `{asset_class_for_label(label)}` in "
            f"`{ASSET_STATES_PATH.name}`."
        )
    else:
        asset_section = "\n".join(
            f"- {title}: **{asset[field]}**" for field, title in ASSET_FIELDS
        )

    return (
        f"### Object {object_id}\n\n"
        f"- Label: **{label}**\n"
        f"- Asset class: **{asset_class_for_label(label)}**\n"
        f"{asset_section}\n\n"
        f"**Geometry**\n\n"
        f"- Highlight points: {len(points):,}\n"
        f"- Center (rendered X, Y, Z): {fmt(points.mean(axis=0))}\n"
        f"- Bounds min: {fmt(points.min(axis=0))}\n"
        f"- Bounds max: {fmt(points.max(axis=0))}"
    )


def main() -> None:
    if not PLY_PATH.is_file():
        raise FileNotFoundError(f"Gaussian PLY file not found: {PLY_PATH}")

    centers, covariances, colors, opacities = load_gaussians(PLY_PATH)
    object_ids, object_points = load_object_points(OBJECT_POINTS_PATH)

    cluster_ids = [int(i) for i in np.unique(object_ids)]
    csv_labels = load_object_labels(SIMILARITIES_PATH)
    labels_by_object = {i: csv_labels.get(i, UNLABELED) for i in cluster_ids}

    # One color per label (objects sharing a label share a color).
    unique_labels = sorted(set(labels_by_object.values()) - {EXCLUDED_LABEL})
    colors_by_label = {
        label: OBJECT_PALETTE[i % len(OBJECT_PALETTE)].astype(np.float32)
        for i, label in enumerate(unique_labels)
    }

    shown_ids = select_object_ids(cluster_ids, OBJECT_IDS_TO_HIGHLIGHT, labels_by_object)

    # Assign over all targeted objects so an object's state doesn't change
    # when OBJECT_IDS_TO_HIGHLIGHT narrows the view.
    targeted_ids = [i for i in cluster_ids if labels_by_object[i] != EXCLUDED_LABEL]
    asset_by_object = assign_asset_states(
        targeted_ids, labels_by_object, load_asset_states(ASSET_STATES_PATH)
    )

    # Everything is rendered with a mirrored Y axis.
    rendered_centers, rendered_covariances = flip_gaussians_y(centers, covariances)
    points_by_object = {
        i: object_points[object_ids == i] * Y_AXIS_FLIP for i in shown_ids
    }

    print(
        f"Loaded {len(centers):,} Gaussian splats and matched "
        f"{len(object_points):,} CSV points across {len(cluster_ids)} objects; "
        f"highlighting {len(shown_ids)} configured objects after "
        f"excluding label '{EXCLUDED_LABEL}'."
    )

    # Camera starts at the scene center, looking along +X with +Y up.
    bounds_min, bounds_max = splat_bounds(rendered_centers, rendered_covariances)
    camera_position = (bounds_min + bounds_max) * 0.5
    camera_look_at = camera_position + np.array([1.0, 0.0, 0.0])
    camera_up = (0.0, 1.0, 0.0)

    server = viser.ViserServer(host="127.0.0.1")
    server.initial_camera.position = camera_position
    server.initial_camera.look_at = camera_look_at
    server.initial_camera.up = camera_up

    selector = server.gui.add_dropdown(
        "Highlight",
        [ALL_OBJECTS, *(f"Object {i}" for i in shown_ids)],
        initial_value=ALL_OBJECTS,
    )
    server.gui.add_markdown(INSTRUCTIONS)
    object_info = server.gui.add_markdown(ALL_OBJECTS_INFO)

    server.scene.add_gaussian_splats(
        "/gaussians", rendered_centers, rendered_covariances, colors, opacities
    )

    highlight_covariance = np.eye(3, dtype=np.float32) * HIGHLIGHT_SIGMA**2
    handles: dict[int, GaussianSplatHandle] = {}
    boxes: dict[int, BoxHandle] = {}

    for object_id, points in points_by_object.items():
        count = len(points)
        color = colors_by_label[labels_by_object[object_id]]

        handles[object_id] = server.scene.add_gaussian_splats(
            f"/highlighted-objects/object-{object_id}",
            points,
            np.tile(highlight_covariance, (count, 1, 1)),
            np.tile(color, (count, 1)),
            np.full((count, 1), HIGHLIGHT_OPACITY, dtype=np.float32),
        )

        low, high = points.min(axis=0), points.max(axis=0)
        extent = high - low
        padding = np.maximum(extent * 0.05, 0.01)
        box = server.scene.add_box(
            f"/object-bounds/object-{object_id}",
            color=color,
            dimensions=np.maximum(extent + 2.0 * padding, 0.02),
            position=(low + high) * 0.5,
            wireframe=True,
            cast_shadow=False,
            receive_shadow=False,
            visible=True,
        )
        boxes[object_id] = box

        @box.on_click
        def _select_from_box(
            _event: SceneNodePointerEvent[BoxHandle], object_id: int = object_id
        ) -> None:
            selector.value = f"Object {object_id}"

    @selector.on_update
    def _update_highlight(_event: object) -> None:
        choice = selector.value
        selected = None if choice == ALL_OBJECTS else int(choice.removeprefix("Object "))

        for object_id in handles:
            visible = selected is None or object_id == selected
            handles[object_id].visible = visible
            boxes[object_id].visible = visible

        if selected is None:
            object_info.content = ALL_OBJECTS_INFO
        else:
            object_info.content = describe_object(
                selected,
                labels_by_object[selected],
                points_by_object[selected],
                asset_by_object.get(selected),
            )

    @server.on_client_connect
    def _initialize_client_camera(client: viser.ClientHandle) -> None:
        client.camera.up_direction = camera_up
        client.camera.position = camera_position
        client.camera.look_at = camera_look_at

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
