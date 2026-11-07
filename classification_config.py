"""Settings for matching detected objects to target classes (PointNet++ similarity)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SIMILARITIES_PATH = PROJECT_DIR / "similarities.csv"

# Cosine similarity needed to accept a match; anything lower is "not in targets".
THRESHOLD = 0.971
# How clouds are reduced to N_POINTS: "uniform", "fps", "voxel" or "random".
SAMPLING = "uniform"
N_POINTS = 4000

# Target class name -> cluster id in the detected-object CSV. Every other cluster is
# matched against these. Cluster ids depend on the DBSCAN settings in
# detection_config.py, so update them if you re-run detection with new settings.
# Use a single integer id per class: the splat viewer reads each target's id from
# similarities.csv (a list of ids would average several clusters but is not supported there).
TARGETS: dict[str, int] = {
    "Valve": 4,
    "Valve_handle": 7,
    "Pressure_sensor": 8,
}
