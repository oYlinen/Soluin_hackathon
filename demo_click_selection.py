"""Demo: click synthetic point-cloud objects in Viser to inspect them."""

import time
from typing import Any, Optional

import numpy as np
import viser

HIGHLIGHT_COLOR = (255, 20, 147)  # Neon magenta
DIM_COLOR = (60, 60, 60)
GROUND_COLOR = (130, 130, 130)
GROUND_DIM_COLOR = (50, 50, 50)
NUM_GROUND_POINTS = 12000
DEFAULT_INFO = (
    "### 🖱️ Click an Object in 3D\n"
    "Click any colored object in the viewport to inspect its description."
)

OBJECTS: dict[str, dict[str, Any]] = {
    "car": {
        "title": "Autonomous Vehicle Chassis",
        "description": "High-density LiDAR scan of an electric vehicle. Detected 4 wheels and rooftop sensor rig.",
        "color": (240, 60, 60),  # Coral red
        "center": np.array([-1.2, -0.5, 0.4]),
        "size": np.array([1.2, 0.7, 0.5]),
        "num_points": 4000,
    },
    "pedestrian": {
        "title": "Standing Pedestrian",
        "description": "Pedestrian standing near the sidewalk curb holding an umbrella.",
        "color": (50, 205, 50),  # Lime green
        "center": np.array([0.8, -0.4, 0.6]),
        "size": np.array([0.4, 0.4, 1.1]),
        "num_points": 2500,
    },
    "traffic_light": {
        "title": "Traffic Light & Signpost",
        "description": "Vertical street infrastructure pole with active green signal and road sign.",
        "color": (255, 190, 30),  # Golden yellow
        "center": np.array([0.2, 1.2, 0.9]),
        "size": np.array([0.3, 0.3, 1.6]),
        "num_points": 2000,
    },
}


def solid(color: tuple[int, int, int], count: int) -> np.ndarray:
    return np.tile(np.array(color, dtype=np.uint8), (count, 1))


def main() -> None:
    server = viser.ViserServer()
    print(f"\nViser is live! Open in your browser: http://localhost:{server.get_port()}\n")

    ground = server.scene.add_point_cloud(
        name="/ground",
        points=np.random.uniform(
            [-2.5, -2.5, 0.0], [2.5, 2.5, 0.04], size=(NUM_GROUND_POINTS, 3)
        ),
        colors=solid(GROUND_COLOR, NUM_GROUND_POINTS),
        point_size=0.01,
    )
    info = server.gui.add_markdown(DEFAULT_INFO)
    reset_button = server.gui.add_button("Reset View & Selection")

    clouds: dict[str, viser.PointCloudHandle] = {}
    selection_label: Optional[viser.LabelHandle] = None
    selection_box: Optional[viser.BoxHandle] = None

    def clear_selection_markers() -> None:
        nonlocal selection_label, selection_box
        for handle in (selection_label, selection_box):
            if handle is not None:
                handle.remove()
        selection_label = selection_box = None

    def select_object(key: str) -> None:
        nonlocal selection_label, selection_box
        obj = OBJECTS[key]
        center, size = obj["center"], obj["size"]

        # Highlight the clicked object, dim everything else.
        ground.colors = solid(GROUND_DIM_COLOR, NUM_GROUND_POINTS)
        for other_key, cloud in clouds.items():
            color = HIGHLIGHT_COLOR if other_key == key else DIM_COLOR
            cloud.colors = solid(color, OBJECTS[other_key]["num_points"])

        clear_selection_markers()
        selection_label = server.scene.add_label(
            name="/active_label",
            text=obj["title"],
            position=(center[0], center[1], center[2] + size[2] / 2.0 + 0.25),
        )
        selection_box = server.scene.add_box(
            name="/active_selection_box",
            position=tuple(center),
            dimensions=tuple(size + 0.08),
            color=HIGHLIGHT_COLOR,
            wireframe=True,
        )
        info.content = (
            f"### 🏷️ {obj['title']}\n"
            f"**Description:**  \n{obj['description']}\n\n"
            "---\n"
            f"* **Object Key:** `{key}`\n"
            f"* **Point Count:** `{obj['num_points']}`\n"
            f"* **3D Center:** `[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]`\n"
            f"* **Dimensions:** `{size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m`\n"
        )

    @reset_button.on_click
    def _reset(_event: object) -> None:
        clear_selection_markers()
        ground.colors = solid(GROUND_COLOR, NUM_GROUND_POINTS)
        for key, cloud in clouds.items():
            cloud.colors = solid(OBJECTS[key]["color"], OBJECTS[key]["num_points"])
        info.content = DEFAULT_INFO

    for key, obj in OBJECTS.items():
        center, size, count = obj["center"], obj["size"], obj["num_points"]

        # Synthetic points inside the object's bounding box (display only).
        clouds[key] = server.scene.add_point_cloud(
            name=f"/objects/{key}/points",
            points=np.random.uniform(center - size / 2.0, center + size / 2.0, size=(count, 3)),
            colors=solid(obj["color"], count),
            point_size=0.014,
        )

        # Translucent box proxy so clicks land on the object's volume.
        proxy = server.scene.add_box(
            name=f"/objects/{key}/click_proxy",
            position=tuple(center),
            dimensions=tuple(size),
            color=obj["color"],
            opacity=0.10,
            wireframe=False,
        )

        @proxy.on_click
        def _on_click(_event: object, key: str = key) -> None:
            select_object(key)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
