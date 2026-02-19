# STAC-group split for CloudSen12: group by scene_id from to_geodataframe(), avoid scene overlap.
# In this TACO packaging we get ~1 patch per scene; each record is the evaluation unit.
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import tacoreader

# Where to save/load split JSON (same seed+ratio => same file)
DEFAULT_SPLIT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "splits")
)


def _get_taco_path(flavor: str = "l1c") -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    name = f"mini-cloudsen12-{flavor}-high-512.taco"
    path = os.path.join(base_dir, "..", "..", "data", "CloudSen12+", "TACOs", name)
    return os.path.normpath(path)


def _normalize_col(gdf, *candidates: str):
    # Match gdf column to one of candidates (case-insensitive, : same as _)
    cols = [c for c in gdf.columns if c is not None]
    for cand in candidates:
        cn = str(cand).lower().replace(":", "_")
        for c in cols:
            if str(c).lower().replace(":", "_") == cn:
                return c
    return None


def _get_scene_ids_from_gdf(gdf) -> List[str]:
    # One scene_id per row. Prefer stac:id, then tile+time, else row index.
    n = len(gdf)
    id_col = _normalize_col(gdf, "stac:id", "id", "stac_id")
    if id_col is not None:
        ids = gdf[id_col].astype(str).tolist()
        return ids

    tile_col = _normalize_col(gdf, "sentinel:tile_id", "tile_id", "sentinel_tile_id")
    time_col = _normalize_col(
        gdf, "stac:time_start", "start_datetime", "datetime", "stac_start_datetime"
    )
    if tile_col is not None and time_col is not None:
        tiles = gdf[tile_col].astype(str)
        times = gdf[time_col].astype(str)
        return [f"{t}_{d}" for t, d in zip(tiles, times)]
    if tile_col is not None:
        return gdf[tile_col].astype(str).tolist()
    if time_col is not None:
        return gdf[time_col].astype(str).tolist()

    return [str(i) for i in range(n)]


def get_scene_groups_from_stac(taco_path: str) -> Tuple[Dict[str, List[int]], List[str]]:
    # scene_id -> list of record indices; and idx_to_scene_id[i] = scene_id for record i.
    dataset = tacoreader.load(taco_path)
    n = len(dataset)

    if not hasattr(dataset, "to_geodataframe"):
        raise AttributeError(
            "tacoreader dataset has no to_geodataframe(); ensure STAC-compliant TACO and tacoreader version."
        )
    gdf = dataset.to_geodataframe()

    if len(gdf) != n:
        raise ValueError(f"GeoDataFrame length {len(gdf)} != dataset length {n}")

    scene_ids = _get_scene_ids_from_gdf(gdf)
    if len(scene_ids) != n:
        scene_ids = scene_ids[:n] if len(scene_ids) >= n else scene_ids + [str(i) for i in range(len(scene_ids), n)]

    scene_groups = defaultdict(list)
    for record_idx in range(n):
        sid = scene_ids[record_idx]
        scene_groups[sid].append(record_idx)

    idx_to_scene_id = scene_ids
    return dict(scene_groups), idx_to_scene_id


def get_scene_split_indices(
    taco_path: str,
    split_ratio: Tuple[float, float, float] = (0.85, 0.05, 0.1),
    seed: int = 42,
    return_summary: bool = False,
    split_dir: Optional[str] = None,
    save_split: bool = True,
):
    # 85/5/10 split over scene groups; save indices, idx_to_scene_id, seed, ratios to JSON.
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("split_ratio must sum to 1.0")

    split_dir = split_dir or DEFAULT_SPLIT_DIR
    split_file = os.path.join(
        split_dir,
        f"cloudsen12_scene_split_s{seed}_r{split_ratio[0]:.2f}_{split_ratio[1]:.2f}_{split_ratio[2]:.2f}.json",
    )

    idx_to_scene_id = None

    if save_split and os.path.isfile(split_file):
        with open(split_file, "r") as f:
            data = json.load(f)
        train_indices = data["train_indices"]
        val_indices = data["val_indices"]
        test_indices = data["test_indices"]
        train_scenes = data["train_scene_ids"]
        val_scenes = data["val_scene_ids"]
        test_scenes = data["test_scene_ids"]
        n_scenes = data.get("n_scenes", len(set(train_scenes + val_scenes + test_scenes)))
        n_patches_total = len(train_indices) + len(val_indices) + len(test_indices)
        idx_to_scene_id = data.get("idx_to_scene_id")
    else:
        scene_groups, idx_to_scene_id = get_scene_groups_from_stac(taco_path)
        scene_ids = list(scene_groups.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(scene_ids)

        n_scenes = len(scene_ids)
        train_r, val_r, test_r = split_ratio
        train_end = int(n_scenes * train_r)
        val_end = train_end + int(n_scenes * val_r)

        train_scenes = scene_ids[:train_end]
        val_scenes = scene_ids[train_end:val_end]
        test_scenes = scene_ids[val_end:]

        train_indices = [i for s in train_scenes for i in scene_groups[s]]
        val_indices = [i for s in val_scenes for i in scene_groups[s]]
        test_indices = [i for s in test_scenes for i in scene_groups[s]]
        n_patches_total = len(train_indices) + len(val_indices) + len(test_indices)

        if save_split:
            os.makedirs(split_dir, exist_ok=True)
            with open(split_file, "w") as f:
                json.dump(
                    {
                        "train_indices": train_indices,
                        "val_indices": val_indices,
                        "test_indices": test_indices,
                        "train_scene_ids": train_scenes,
                        "val_scene_ids": val_scenes,
                        "test_scene_ids": test_scenes,
                        "idx_to_scene_id": idx_to_scene_id,
                        "n_scenes": n_scenes,
                        "split_ratio": list(split_ratio),
                        "seed": seed,
                    },
                    f,
                    indent=0,
                )

    if not return_summary:
        return train_indices, val_indices, test_indices

    if idx_to_scene_id is None:
        _, idx_to_scene_id = get_scene_groups_from_stac(taco_path)

    train_set = set(train_scenes)
    val_set = set(val_scenes)
    test_set = set(test_scenes)
    no_leakage = (
        len(train_set & val_set) == 0
        and len(train_set & test_set) == 0
        and len(val_set & test_set) == 0
    )
    scene_groups_counts = []
    if idx_to_scene_id is not None:
        sid_to_count = Counter(idx_to_scene_id)
        scene_groups_counts = list(sid_to_count.values())
    min_pps = min(scene_groups_counts) if scene_groups_counts else 0
    max_pps = max(scene_groups_counts) if scene_groups_counts else 0
    median_pps = int(np.median(scene_groups_counts)) if scene_groups_counts else 0

    summary = {
        "n_patches_total": n_patches_total,
        "n_scenes": n_scenes,
        "n_scenes_train": len(train_scenes),
        "n_scenes_val": len(val_scenes),
        "n_scenes_test": len(test_scenes),
        "n_records_train": len(train_indices),
        "n_records_val": len(val_indices),
        "n_records_test": len(test_indices),
        "no_leakage": no_leakage,
        "split_ratio": split_ratio,
        "seed": seed,
        "split_file": split_file if save_split else None,
        "idx_to_scene_id": idx_to_scene_id,
        "min_patches_per_scene": min_pps,
        "median_patches_per_scene": median_pps,
        "max_patches_per_scene": max_pps,
    }
    return train_indices, val_indices, test_indices, summary


def get_scene_groups_and_idx_to_scene_id(taco_path: str) -> Tuple[Dict[str, List[int]], List[str]]:
    return get_scene_groups_from_stac(taco_path)


def format_split_summary(summary: dict) -> str:
    n_train = summary["n_records_train"]
    n_val = summary["n_records_val"]
    n_test = summary["n_records_test"]
    n_scenes = summary["n_scenes"]
    n_patches = summary["n_patches_total"]
    min_pps = summary.get("min_patches_per_scene", 0)
    median_pps = summary.get("median_patches_per_scene", 0)
    max_pps = summary.get("max_patches_per_scene", 0)

    lines = [
        "[STAC-group split (nearly patch-wise in this packaging)]",
        f"  total records (patches) = {n_patches}",
        f"  unique scenes = {n_scenes}  (min/median/max patches per scene: {min_pps}/{median_pps}/{max_pps})",
        f"  train: {summary['n_scenes_train']} scenes, {n_train} patches",
        f"  val:   {summary['n_scenes_val']} scenes, {n_val} patches",
        f"  test:  {summary['n_scenes_test']} scenes, {n_test} patches",
        f"  No scene overlap across splits: {summary['no_leakage']}",
    ]
    if summary.get("split_file"):
        lines.append(f"  Split saved to: {summary['split_file']}")
    return "\n".join(lines)
