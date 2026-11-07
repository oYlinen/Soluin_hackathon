#!/usr/bin/env python3
"""Entry point for the Equipment Room object-detection and highlight tools.

Usage:
    python main.py              # same as `python main.py all`
    python main.py all          # detect (if needed), then open both highlight viewers
    python main.py detect       # cluster Gaussians with DBSCAN and save the object CSV
    python main.py pointcloud   # Open3D point-cloud viewer with object highlights
    python main.py splats       # Viser Gaussian-splat viewer with labeled highlights
    python main.py demo         # synthetic click-to-inspect demo (no data files needed)

Options:
    all --force-detect          # re-run detection even if the object CSV exists
    detect --headless           # detect and save the CSV without opening a viewer
"""

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

# command -> (module with a main() function, help text)
COMMANDS: dict[str, tuple[str, str]] = {
    "detect": ("detect_objects", "detect objects with DBSCAN and save them to CSV"),
    "pointcloud": ("view_pointcloud_highlight", "highlight detected objects in the point cloud (Open3D)"),
    "splats": ("view_splat_highlight", "highlight labeled objects on the Gaussian splats (Viser)"),
    "demo": ("demo_click_selection", "run the synthetic click-to-inspect demo"),
}
THIS_FILE = Path(__file__).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    all_parser = subparsers.add_parser(
        "all", help="detect objects if needed, then open both highlight viewers"
    )
    all_parser.add_argument(
        "--force-detect", action="store_true",
        help="re-run object detection even if the CSV already exists",
    )
    for name, (_module, help_text) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        if name == "detect":
            sub.add_argument(
                "--headless", action="store_true",
                help="detect and save the CSV without opening the viewer",
            )
    return parser


def run_all(force_detect: bool) -> None:
    """Detect (if needed), then run the splat and point-cloud viewers together."""
    from detection_config import OBJECT_POINTS_PATH

    if force_detect or not OBJECT_POINTS_PATH.is_file():
        print("Step 1/2: detecting objects...")
        from detect_objects import detect_and_save

        if detect_and_save() == 0:
            raise SystemExit(
                "No objects were found; adjust DBSCAN_EPS / DBSCAN_MIN_POINTS "
                "in detection_config.py."
            )
    else:
        print(
            f"Step 1/2: using existing {OBJECT_POINTS_PATH.name} "
            "(use --force-detect to redo detection)."
        )

    # Each viewer runs in its own process: Open3D's GUI must own its main thread
    # and the Viser viewer blocks forever.
    print("Step 2/2: launching viewers. Close the Open3D window or press Ctrl+C to stop.")
    splats = subprocess.Popen([sys.executable, str(THIS_FILE), "splats"])
    cloud = subprocess.Popen([sys.executable, str(THIS_FILE), "pointcloud"])
    splats_warned = False
    try:
        while cloud.poll() is None:
            if splats.poll() is not None and not splats_warned:
                print(f"Splat viewer exited with code {splats.returncode}.")
                splats_warned = True
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (splats, cloud):
            if process.poll() is None:
                process.terminate()
        for process in (splats, cloud):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command is None:  # plain `python main.py` runs everything
        args.command, args.force_detect = "all", False

    if args.command == "all":
        run_all(args.force_detect)
    elif args.command == "detect" and args.headless:
        from detect_objects import detect_and_save

        detect_and_save()
    else:
        module_name, _ = COMMANDS[args.command]
        # Imported lazily so each command only loads the heavy libraries it needs.
        importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()
