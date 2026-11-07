#!/usr/bin/env python3
"""Entry point for the Equipment Room object-detection and highlight tools.

Usage:
    python main.py              # same as `python main.py all`
    python main.py all          # detect -> classify -> open the Gaussian-splat viewer
    python main.py detect       # cluster Gaussians with DBSCAN and save the object CSV
    python main.py classify     # match detected objects to targets, write similarities.csv
    python main.py pointcloud   # Open3D point-cloud viewer with object highlights
    python main.py splats       # Viser Gaussian-splat viewer with labeled highlights
    python main.py demo         # synthetic click-to-inspect demo (no data files needed)

Options:
    all --force-detect          # re-run detection (and classification) even if the CSVs exist
    all --force-classify        # re-run classification even if similarities.csv exists
    detect --headless           # detect and save the CSV without opening a viewer
"""

import argparse
import importlib

# command -> (module with a main() function, help text)
COMMANDS: dict[str, tuple[str, str]] = {
    "detect": ("detect_objects", "detect objects with DBSCAN and save them to CSV"),
    "classify": ("classify_objects", "match detected objects to target classes (writes similarities.csv)"),
    "pointcloud": ("view_pointcloud_highlight", "highlight detected objects in the point cloud (Open3D)"),
    "splats": ("view_splat_highlight", "highlight labeled objects on the Gaussian splats (Viser)"),
    "demo": ("demo_click_selection", "run the synthetic click-to-inspect demo"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    all_parser = subparsers.add_parser(
        "all", help="detect objects, classify them, then open the Gaussian-splat viewer"
    )
    all_parser.add_argument(
        "--force-detect", action="store_true",
        help="re-run detection (and therefore classification) even if the CSV already exists",
    )
    all_parser.add_argument(
        "--force-classify", action="store_true",
        help="re-run classification even if similarities.csv already exists",
    )
    for name, (_module, help_text) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        if name == "detect":
            sub.add_argument(
                "--headless", action="store_true",
                help="detect and save the CSV without opening the viewer",
            )
    return parser


def run_all(force_detect: bool = False, force_classify: bool = False) -> None:
    """Detect objects, classify them, then open the Gaussian-splat viewer.

    Each of the first two steps is skipped when its output file already exists,
    unless forced. Fresh detection always triggers classification, since the
    cluster IDs in similarities.csv would no longer match the new object CSV.
    """
    from classification_config import SIMILARITIES_PATH
    from detection_config import OBJECT_POINTS_PATH

    need_detect = force_detect or not OBJECT_POINTS_PATH.is_file()
    need_classify = need_detect or force_classify or not SIMILARITIES_PATH.is_file()

    if need_detect:
        print("Step 1/3: detecting objects...")
        from detect_objects import detect_and_save

        if detect_and_save() == 0:
            raise SystemExit(
                "No objects were found; adjust DBSCAN_EPS / DBSCAN_MIN_POINTS "
                "in detection_config.py."
            )
    else:
        print(
            f"Step 1/3: using existing {OBJECT_POINTS_PATH.name} "
            "(use --force-detect to redo detection)."
        )

    if need_classify:
        print("Step 2/3: classifying objects against the target classes...")
        importlib.import_module(COMMANDS["classify"][0]).main()
    else:
        print(
            f"Step 2/3: using existing {SIMILARITIES_PATH.name} "
            "(use --force-classify to redo classification)."
        )

    print("Step 3/3: opening the Gaussian-splat viewer.")
    importlib.import_module(COMMANDS["splats"][0]).main()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command is None:  # plain `python main.py` runs everything
        args.command, args.force_detect, args.force_classify = "all", False, False

    if args.command == "all":
        run_all(args.force_detect, args.force_classify)
    elif args.command == "detect" and args.headless:
        from detect_objects import detect_and_save

        detect_and_save()
    else:
        module_name, _ = COMMANDS[args.command]
        # Imported lazily so each command only loads the heavy libraries it needs.
        importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()
