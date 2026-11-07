"""DBSCAN settings and the resulting detected-object CSV path."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DBSCAN_EPS = 0.1
DBSCAN_MIN_POINTS = 800
OBJECT_POINTS_PATH = PROJECT_DIR / (
    f"detected_object_points_eps_{DBSCAN_EPS:g}_min_points_{DBSCAN_MIN_POINTS}.csv"
)
