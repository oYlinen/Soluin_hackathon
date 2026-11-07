#!/usr/bin/env python3
"""Open3D viewer that highlights detected objects in the PLY point cloud."""

import numpy as np
import open3d as o3d

from detection_config import OBJECT_POINTS_PATH
from scene_data import OBJECT_PALETTE, PLY_PATH, Y_AXIS_FLIP, load_object_points

POINT_KEY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
BACKGROUND_COLOR = [0.22, 0.24, 0.28]
ALL_OBJECTS = "All detected objects"


def _point_keys(points: np.ndarray) -> np.ndarray:
    """View (N, 3) float32 points as 1-D structured keys for exact matching."""
    return np.ascontiguousarray(points, dtype=np.float32).view(POINT_KEY_DTYPE).reshape(-1)


def match_objects_to_cloud(
    cloud_points: np.ndarray,
    object_ids: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    """Per-cloud-point object ID (-1 where the point belongs to no object)."""
    unique_keys, point_to_unique = np.unique(_point_keys(cloud_points), return_inverse=True)
    object_keys = _point_keys(object_points)

    unique_indices = np.minimum(np.searchsorted(unique_keys, object_keys), len(unique_keys) - 1)
    unmatched = unique_keys[unique_indices] != object_keys
    if unmatched.any():
        raise ValueError(
            f"{np.count_nonzero(unmatched)} CSV points do not match the PLY coordinates"
        )

    # A coordinate may repeat, but only with the same object ID.
    pairs = np.unique(np.column_stack((unique_indices, object_ids)), axis=0)
    if len(np.unique(pairs[:, 0])) != len(pairs):
        raise ValueError("The CSV assigns the same point coordinate to multiple objects")

    unique_object_ids = np.full(len(unique_keys), -1, dtype=np.int32)
    unique_object_ids[pairs[:, 0]] = pairs[:, 1]
    return unique_object_ids[point_to_unique]


def flip_y_axis(points: np.ndarray) -> np.ndarray:
    return points * Y_AXIS_FLIP


def colors_for_ids(point_ids: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    """Palette color per point; an object keeps its color whatever is selected."""
    return OBJECT_PALETTE[np.searchsorted(cluster_ids, point_ids) % len(OBJECT_PALETTE)]


def main() -> None:
    if not PLY_PATH.is_file():
        raise FileNotFoundError(f"PLY point cloud not found: {PLY_PATH}")

    print(f"Loading point cloud: {PLY_PATH}")
    point_cloud = o3d.io.read_point_cloud(str(PLY_PATH))
    if point_cloud.is_empty():
        raise ValueError(f"No points could be loaded from {PLY_PATH}")

    object_ids, object_points = load_object_points(OBJECT_POINTS_PATH)
    point_object_ids = match_objects_to_cloud(
        np.asarray(point_cloud.points), object_ids, object_points
    )
    cloud_points = flip_y_axis(np.asarray(point_cloud.points))
    point_cloud.points = o3d.utility.Vector3dVector(cloud_points)
    cluster_ids = np.unique(object_ids)  # sorted
    print(
        f"Loaded {len(cloud_points):,} PLY points and matched "
        f"{len(object_points):,} CSV points across {len(cluster_ids)} objects."
    )

    # --- Window and scene -------------------------------------------------
    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering
    app = gui.Application.instance
    app.initialize()
    window = app.create_window("PLY Object Highlighter", 1280, 800)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.08, 0.08, 0.08, 1.0])

    background = o3d.geometry.PointCloud()
    background.points = point_cloud.points
    background.paint_uniform_color(BACKGROUND_COLOR)
    background_material = rendering.MaterialRecord()
    background_material.shader = "defaultUnlit"
    background_material.point_size = 2.0
    scene_widget.scene.add_geometry("background", background, background_material)

    overlay_name = "highlighted_objects"
    overlay_material = rendering.MaterialRecord()
    overlay_material.shader = "defaultUnlit"
    overlay_material.point_size = 5.0

    # --- Controls ---------------------------------------------------------
    font_size = window.theme.font_size
    controls = gui.Vert(0.5 * font_size, gui.Margins(font_size))
    controls.add_child(gui.Label("Choose an object to highlight"))
    selector = gui.Combobox()
    selector.add_item(ALL_OBJECTS)
    for cluster_id in cluster_ids:
        selector.add_item(f"Object {cluster_id}")
    controls.add_child(selector)
    status = gui.Label(f"{len(cluster_ids)} detected objects")
    controls.add_child(status)
    window.add_child(controls)
    window.add_child(scene_widget)

    def show_selection(label: str, _index: int) -> None:
        if scene_widget.scene.has_geometry(overlay_name):
            scene_widget.scene.remove_geometry(overlay_name)

        if label == ALL_OBJECTS:
            mask = point_object_ids >= 0
            object_count = len(cluster_ids)
        else:
            mask = point_object_ids == int(label.removeprefix("Object "))
            object_count = 1

        if mask.any():
            overlay = o3d.geometry.PointCloud()
            overlay.points = o3d.utility.Vector3dVector(cloud_points[mask])
            overlay.colors = o3d.utility.Vector3dVector(
                colors_for_ids(point_object_ids[mask], cluster_ids)
            )
            scene_widget.scene.add_geometry(overlay_name, overlay, overlay_material)
            status.text = f"Showing {object_count} object(s), {int(mask.sum()):,} points"
        else:
            status.text = "No points found for this selection"
        window.post_redraw()

    def layout(_context: gui.LayoutContext) -> None:
        rect = window.content_rect
        panel_width = min(320, max(240, rect.width // 4))
        controls.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        scene_widget.frame = gui.Rect(
            rect.x + panel_width, rect.y, rect.width - panel_width, rect.height
        )

    selector.set_on_selection_changed(show_selection)
    selector.selected_index = 0
    show_selection(ALL_OBJECTS, 0)
    window.set_on_layout(layout)
    scene_widget.setup_camera(
        60.0,
        point_cloud.get_axis_aligned_bounding_box(),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    app.run()


if __name__ == "__main__":
    main()
