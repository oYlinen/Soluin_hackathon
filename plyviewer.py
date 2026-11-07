import time
from typing import Dict, Any, Optional
import numpy as np
import viser

# ---------------------------------------------------------------------------
# 1. Define Scene Objects and Descriptions
# ---------------------------------------------------------------------------
objects_metadata: Dict[str, Dict[str, Any]] = {
    "car": {
        "title": "Autonomous Vehicle Chassis",
        "description": "High-density LiDAR scan of an electric vehicle. Detected 4 wheels and rooftop sensor rig.",
        "color": [240, 60, 60],     # Coral Red
        "center": np.array([-1.2, -0.5, 0.4]),
        "size": np.array([1.2, 0.7, 0.5]),
        "num_points": 4000,
    },
    "pedestrian": {
        "title": "Standing Pedestrian",
        "description": "Pedestrian standing near the sidewalk curb holding an umbrella.",
        "color": [50, 205, 50],     # Lime Green
        "center": np.array([0.8, -0.4, 0.6]),
        "size": np.array([0.4, 0.4, 1.1]),
        "num_points": 2500,
    },
    "traffic_light": {
        "title": "Traffic Light & Signpost",
        "description": "Vertical street infrastructure pole with active green signal and road sign.",
        "color": [255, 190, 30],    # Golden Yellow
        "center": np.array([0.2, 1.2, 0.9]),
        "size": np.array([0.3, 0.3, 1.6]),
        "num_points": 2000,
    },
}

# ---------------------------------------------------------------------------
# 2. Start Viser Server
# ---------------------------------------------------------------------------
server = viser.ViserServer()
print(f"\nViser is live! Open in your browser: http://localhost:{server.get_port()}\n")

# Add Ground Plane (Environment background)
num_ground = 12000
ground_pts = np.random.uniform([-2.5, -2.5, 0.0], [2.5, 2.5, 0.04], size=(num_ground, 3))
ground_pcd = server.scene.add_point_cloud(
    name="/ground",
    points=ground_pts,
    colors=np.full((num_ground, 3), [130, 130, 130], dtype=np.uint8),
    point_size=0.01,
)

# ---------------------------------------------------------------------------
# 3. Sidebar GUI Setup
# ---------------------------------------------------------------------------
gui_info = server.gui.add_markdown(
    content="### 🖱️ Click an Object in 3D\nClick any colored object in the viewport to inspect its description."
)

btn_reset = server.gui.add_button("Reset View & Selection")

pcd_handles = {}
proxy_boxes = {}
active_label_handle: Optional[viser.LabelHandle] = None
active_box_handle: Optional[viser.BoxHandle] = None

# ---------------------------------------------------------------------------
# 4. Selection & Text Display Handler
# ---------------------------------------------------------------------------
def select_object(obj_key: str):
    global active_label_handle, active_box_handle

    obj = objects_metadata[obj_key]
    center = obj["center"]
    size = obj["size"]

    # 1. Highlight clicked object points, dim the rest
    ground_pcd.colors = np.full((num_ground, 3), [50, 50, 50], dtype=np.uint8)
    for key, handle in pcd_handles.items():
        if key == obj_key:
            handle.colors = np.full((objects_metadata[key]["num_points"], 3), [255, 20, 147], dtype=np.uint8) # Neon Magenta
        else:
            handle.colors = np.full((objects_metadata[key]["num_points"], 3), [60, 60, 60], dtype=np.uint8)   # Dimmed

    # 2. Remove previous 3D text label and selection box if they exist
    if active_label_handle is not None:
        active_label_handle.remove()
        active_label_handle = None
    if active_box_handle is not None:
        active_box_handle.remove()
        active_box_handle = None

    # 3. Create 3D floating text label directly above the object
    label_position = (center[0], center[1], center[2] + (size[2] / 2.0) + 0.25)
    active_label_handle = server.scene.add_label(
        name="/active_label",
        text=obj["title"],
        position=label_position,
    )

    # 4. Draw bright wireframe bounding box around the selected object
    active_box_handle = server.scene.add_box(
        name="/active_selection_box",
        position=tuple(center),
        dimensions=tuple(size + 0.08),
        color=(255, 20, 147),
        wireframe=True,
    )

    # 5. Update the sidebar text with rich description
    gui_info.content = f"""
### 🏷️ {obj['title']}
**Description:**  
{obj['description']}

---
* **Object Key:** `{obj_key}`
* **Point Count:** `{obj['num_points']}`
* **3D Center:** `[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]`
* **Dimensions:** `{size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m`
"""

def reset_all():
    global active_label_handle, active_box_handle

    if active_label_handle is not None:
        active_label_handle.remove()
        active_label_handle = None
    if active_box_handle is not None:
        active_box_handle.remove()
        active_box_handle = None

    ground_pcd.colors = np.full((num_ground, 3), [130, 130, 130], dtype=np.uint8)
    for key, handle in pcd_handles.items():
        handle.colors = np.full((objects_metadata[key]["num_points"], 3), objects_metadata[key]["color"], dtype=np.uint8)
    
    gui_info.content = "### 🖱️ Click an Object in 3D\nClick any colored object in the viewport to inspect its description."

btn_reset.on_click(lambda _: reset_all())

# ---------------------------------------------------------------------------
# 5. Populate Objects & Register Click Callbacks
# ---------------------------------------------------------------------------
for key, obj in objects_metadata.items():
    center = obj["center"]
    size = obj["size"]
    n_pts = obj["num_points"]

    # Generate synthetic points within the bounding box
    pts = np.random.uniform(center - size / 2.0, center + size / 2.0, size=(n_pts, 3))
    colors = np.full((n_pts, 3), obj["color"], dtype=np.uint8)

    # 1. Add point cloud to the scene (display only)
    pcd = server.scene.add_point_cloud(
        name=f"/objects/{key}/points",
        points=pts,
        colors=colors,
        point_size=0.014,
    )
    pcd_handles[key] = pcd

    # 2. Add clickable box proxy covering the object's volume
    box = server.scene.add_box(
        name=f"/objects/{key}/click_proxy",
        position=tuple(center),
        dimensions=tuple(size),
        color=tuple(obj["color"]),
        opacity=0.10,       # Low opacity so the points inside are clearly visible
        wireframe=False,
    )
    proxy_boxes[key] = box

    # 3. Register click callback on the box proxy
    def register_click(box_handle, target_key):
        @box_handle.on_click
        def _(event):
            select_object(target_key)

    register_click(box, key)

# Keep the server running
while True:
    time.sleep(1.0)