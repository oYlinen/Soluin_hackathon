#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import random

# Load the point cloud
pcd = o3d.io.read_point_cloud("Equipment Room.ply")
# Fixed the missing method: cluster_dbscan
labels = pcd.cluster_dbscan(eps=0.05, min_points=200, print_progress=False)

labels_np = np.array(labels)
max_label = labels_np.max()

object_list = []

# Loop through each cluster and color them randomly
for i in range(max_label + 1):
    # Fast filtering using NumPy instead of Python list comprehension
    obj_indices = np.where(labels_np == i)[0]
    individual_object = pcd.select_by_index(obj_indices)

    # Generate a random RGB color for this object
    random_color = [random.random(), random.random(), random.random()]
    individual_object.paint_uniform_color(random_color)

    object_list.append(individual_object)

print(f"Found: {len(object_list)} objects")

for idx, obj in enumerate(object_list):
    continue
    # Avaa ikkunan tälle yksittäiselle objektille
    o3d.visualization.draw_geometries([obj],
                                      window_name=f"{idx}",
                                      width=800, height=600)




