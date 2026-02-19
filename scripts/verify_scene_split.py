#!/usr/bin/env python3
# Check STAC-group split: counts, patches per scene, no scene in two splits. Run from project root.
import argparse
import os
import sys

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders.cloudsen12_scene_split import (
    _get_taco_path,
    format_split_summary,
    get_scene_groups_from_stac,
    get_scene_split_indices,
)


def parse_split_ratio(s):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--split-ratio must be three numbers, e.g. 0.85,0.05,0.1")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Verify scene-level train/val/test split")
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.85,0.05,0.1",
        help="Train,val,test fractions (e.g. 0.85,0.05,0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--taco",
        choices=["l1c", "l2a"],
        default="l1c",
        help="Which TACO to use (same record order for both)",
    )
    args = parser.parse_args()

    split_ratio = parse_split_ratio(args.split_ratio)
    taco_path = _get_taco_path(args.taco)

    if not os.path.isfile(taco_path):
        print(f"TACO not found: {taco_path}")
        print("Run from project root and ensure data/CloudSen12+/TACOs/ exists.")
        sys.exit(1)

    print("STAC-group split (nearly patch-wise in this packaging).")
    print("Building scene -> patch indices from TACO (to_geodataframe)...")
    try:
        scene_groups, _ = get_scene_groups_from_stac(taco_path)
    except AttributeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    n_patches = sum(len(v) for v in scene_groups.values())
    n_scenes = len(scene_groups)
    scene_ids = list(scene_groups.keys())
    print(f"  total records (patches) = {n_patches}")
    print(f"  unique scenes = {n_scenes}")
    if n_scenes < n_patches:
        print(f"  -> Multiple patches per scene (avg {n_patches / n_scenes:.1f} patches/scene)")
    elif n_scenes >= n_patches * 0.9:
        print("  WARNING: unique scenes ~= patches; scene_id logic may be wrong.")
    sample = scene_ids[:5]
    print(f"  Sample scene IDs: {sample}")
    print()

    print("Applying scene-level split...")
    train_indices, val_indices, test_indices, summary = get_scene_split_indices(
        taco_path, split_ratio=split_ratio, seed=args.seed, return_summary=True
    )

    print(format_split_summary(summary))
    print()

    if not summary["no_leakage"]:
        print("WARNING: At least one scene appears in more than one split!")
        sys.exit(1)
    print("OK: no scene in more than one split.")


if __name__ == "__main__":
    main()
