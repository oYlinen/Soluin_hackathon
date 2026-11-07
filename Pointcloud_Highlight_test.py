#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import open3d as o3d

from object_detection_config import OBJECT_POINTS_PATH

PROJECT_DIR = Path(__file__).resolve().parent
PLY_PATH = PROJECT_DIR.parent / "Equipment Room.ply"
POINT_KEY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
BACKGROUND_COLOR = np.array([0.22, 0.24, 0.28], dtype=np.float64)
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


def load_object_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Detected-object CSV not found: {path}. Run test_object_detection.py "
            "and click Detect objects first."
        )

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0 or data.dtype.names is None:
        raise ValueError(f"{path} contains no detected object points")

    required_columns = {"cluster_id", "x", "y", "z"}
    missing_columns = required_columns.difference(data.dtype.names)
    if missing_columns:
        raise ValueError(
            f"{path} is missing CSV columns: {', '.join(sorted(missing_columns))}"
        )

    data = np.atleast_1d(data)
    cluster_ids = np.asarray(data["cluster_id"], dtype=np.float64)
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    if not np.isfinite(cluster_ids).all() or not np.isfinite(points).all():
        raise ValueError(f"{path} contains non-finite cluster IDs or coordinates")
    if not np.equal(cluster_ids, np.floor(cluster_ids)).all():
        raise ValueError(f"{path} contains non-integer cluster IDs")
    return cluster_ids.astype(np.int32), points


def match_objects_to_cloud(
    cloud_points: np.ndarray,
    object_ids: np.ndarray,
    object_points: np.ndarray,
) -> np.ndarray:
    point_keys = np.ascontiguousarray(cloud_points, dtype=np.float32).view(
        POINT_KEY_DTYPE
    ).reshape(-1)
    unique_keys, point_to_unique = np.unique(point_keys, return_inverse=True)
    object_keys = np.ascontiguousarray(object_points, dtype=np.float32).view(
        POINT_KEY_DTYPE
    ).reshape(-1)

    unique_indices = np.searchsorted(unique_keys, object_keys)
    if np.any(unique_indices >= len(unique_keys)):
        raise ValueError("Some CSV points do not match any point in the PLY file")
    matched = unique_keys[unique_indices] == object_keys
    if not np.all(matched):
        raise ValueError(
            f"{np.count_nonzero(~matched)} CSV points do not match the PLY coordinates"
        )

    unique_object_ids = np.full(len(unique_keys), -1, dtype=np.int32)
    for unique_index, object_id in zip(unique_indices, object_ids):
        existing_id = unique_object_ids[unique_index]
        if existing_id not in (-1, object_id):
            raise ValueError(
                "The CSV assigns the same point coordinate to multiple objects"
            )
        unique_object_ids[unique_index] = object_id
    return unique_object_ids[point_to_unique]


def flip_y_axis(points: np.ndarray) -> np.ndarray:
    rendered_points = points.copy()
    rendered_points[:, 1] *= -1
    return rendered_points


def main() -> None:
    if not PLY_PATH.is_file():
        raise FileNotFoundError(f"PLY point cloud not found: {PLY_PATH}")

    print(f"Loading point cloud: {PLY_PATH}")
    point_cloud = o3d.io.read_point_cloud(str(PLY_PATH))
    if point_cloud.is_empty():
        raise ValueError(f"No points could be loaded from {PLY_PATH}")

    object_ids, object_points = load_object_points(OBJECT_POINTS_PATH)
    cloud_points = np.asarray(point_cloud.points)
    point_object_ids = match_objects_to_cloud(cloud_points, object_ids, object_points)
    cloud_points = flip_y_axis(cloud_points)
    point_cloud.points = o3d.utility.Vector3dVector(cloud_points)
    cluster_ids = np.unique(object_ids)
    print(
        f"Loaded {len(cloud_points):,} PLY points and matched "
        f"{len(object_points):,} CSV points across {len(cluster_ids)} objects."
    )
    gui = o3d.visualization.gui
    app = gui.Application.instance
    app.initialize()
    window = app.create_window("PLY Object Highlighter", 1280, 800)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.08, 0.08, 0.08, 1.0])

    background = o3d.geometry.PointCloud()
    background.points = point_cloud.points
    background.paint_uniform_color(BACKGROUND_COLOR.tolist())
    background_material = o3d.visualization.rendering.MaterialRecord()
    background_material.shader = "defaultUnlit"
    background_material.point_size = 2.0
    scene_widget.scene.add_geometry("background", background, background_material)

    controls = gui.Vert(0.5 * window.theme.font_size, gui.Margins(1.0 * window.theme.font_size))
    controls.add_child(gui.Label("Choose an object to highlight"))
    selector = gui.Combobox()
    selector.add_item("All detected objects")
    for cluster_id in cluster_ids:
        selector.add_item(f"Object {cluster_id}")
    controls.add_child(selector)
    status = gui.Label(f"{len(cluster_ids)} detected objects")
    controls.add_child(status)
    window.add_child(controls)
    window.add_child(scene_widget)

    overlay_name = "highlighted_objects"
    overlay_material = o3d.visualization.rendering.MaterialRecord()
    overlay_material.shader = "defaultUnlit"
    overlay_material.point_size = 5.0

    def show_selection(label: str, _index: int) -> None:
        if scene_widget.scene.has_geometry(overlay_name):
            scene_widget.scene.remove_geometry(overlay_name)

        if label == "All detected objects":
            selected_mask = point_object_ids >= 0
            selected_ids = point_object_ids[selected_mask]
            selected_points = cloud_points[selected_mask]
            selected_clusters = cluster_ids
        else:
            cluster_id = int(label.removeprefix("Object "))
            selected_mask = point_object_ids == cluster_id
            selected_ids = point_object_ids[selected_mask]
            selected_points = cloud_points[selected_mask]
            selected_clusters = np.array([cluster_id])

        if len(selected_points):
            overlay = o3d.geometry.PointCloud()
            overlay.points = o3d.utility.Vector3dVector(selected_points)
            colors = np.empty((len(selected_ids), 3), dtype=np.float64)
            for palette_index, cluster_id in enumerate(selected_clusters):
                colors[selected_ids == cluster_id] = OBJECT_PALETTE[
                    palette_index % len(OBJECT_PALETTE)
                ]
            overlay.colors = o3d.utility.Vector3dVector(colors)
            scene_widget.scene.add_geometry(overlay_name, overlay, overlay_material)
            status.text = (
                f"Showing {len(selected_clusters)} object(s), "
                f"{len(selected_points):,} points"
            )
        else:
            status.text = "No points found for this selection"
        window.post_redraw()

    selector.set_on_selection_changed(show_selection)
    selector.selected_index = 0
    show_selection("All detected objects", 0)

    def layout(_context: o3d.visualization.gui.LayoutContext) -> None:
        rect = window.content_rect
        panel_width = min(320, max(240, rect.width // 4))
        controls.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        scene_widget.frame = gui.Rect(
            rect.x + panel_width,
            rect.y,
            rect.width - panel_width,
            rect.height,
        )

    window.set_on_layout(layout)
    scene_widget.setup_camera(
        60.0,
        point_cloud.get_axis_aligned_bounding_box(),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    app.run()


if __name__ == "__main__":
    main()