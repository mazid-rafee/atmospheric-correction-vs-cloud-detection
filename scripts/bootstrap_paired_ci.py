#!/usr/bin/env python3
# Paired bootstrap 95% CI for L2A − L1C: per-class and mIoU. Align by record_idx; drop NaNs per metric.
# Optional: --per-scene samples scene IDs and averages per-scene d_i.
# Reads per_record_iou_l1c.csv, per_record_iou_l2a.csv; writes outputs/stats/paired_bootstrap_ci.json
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")
STATS_DIR = os.path.join(PROJECT_ROOT, "outputs", "stats")
CSV_L1C = os.path.join(METRICS_DIR, "per_record_iou_l1c.csv")
CSV_L2A = os.path.join(METRICS_DIR, "per_record_iou_l2a.csv")
OUT_JSON = os.path.join(STATS_DIR, "paired_bootstrap_ci.json")

CLASS_LABELS = {0: "clear", 1: "thick_cloud", 2: "thin", 3: "cloud_shadow"}


def bootstrap_paired_ci(d_l1c, d_l2a, B=2000, seed=42):
    """Paired difference d = L2A - L1C; drop NaNs; bootstrap mean(d). Returns mean_delta, ci_low, ci_high, n_valid."""
    d = np.asarray(d_l2a, dtype=float) - np.asarray(d_l1c, dtype=float)
    valid = ~np.isnan(d)
    d_valid = d[valid]
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    n = len(d_valid)
    means = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(d_valid[idx]))
    means = np.array(means)
    mean_delta = float(np.mean(d_valid))
    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    return mean_delta, low, high, n_valid


def bootstrap_paired_per_scene(df, col_l1c, col_l2a, B=2000, seed=42):
    """Sample scene_id with replacement; for each scene use mean(d_i) over records; bootstrap mean of those."""
    df = df.copy()
    df["d"] = df[col_l2a].astype(float) - df[col_l1c].astype(float)
    # per-scene mean of d (drop NaN within scene)
    scene_means = df.groupby("scene_id")["d"].apply(lambda x: np.nanmean(x)).reset_index()
    scene_means = scene_means.rename(columns={"d": "mean_d"})
    d_vals = scene_means["mean_d"].values
    valid = ~np.isnan(d_vals)
    d_valid = d_vals[valid]
    if len(d_valid) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    n = len(d_valid)
    means = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(d_valid[idx]))
    means = np.array(means)
    mean_delta = float(np.mean(d_valid))
    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    return mean_delta, low, high, int(valid.sum())


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap 95% CI for L2A − L1C (per-class and mIoU)")
    parser.add_argument("--B", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l1c-csv", default=CSV_L1C)
    parser.add_argument("--l2a-csv", default=CSV_L2A)
    parser.add_argument("--out", default=OUT_JSON)
    parser.add_argument("--per-scene", action="store_true", help="Also run bootstrap over scenes (sample scene IDs)")
    args = parser.parse_args()

    if not os.path.isfile(args.l1c_csv) or not os.path.isfile(args.l2a_csv):
        print(f"Run compute_per_record_iou.py first to create {args.l1c_csv} and {args.l2a_csv}")
        sys.exit(1)

    df_l1c = pd.read_csv(args.l1c_csv)
    df_l2a = pd.read_csv(args.l2a_csv)
    common = np.intersect1d(df_l1c["record_idx"].values, df_l2a["record_idx"].values)
    df_l1c = df_l1c.set_index("record_idx").loc[common].reset_index()
    df_l2a = df_l2a.set_index("record_idx").loc[common].reset_index()
    assert len(df_l1c) == len(df_l2a)
    n_records = len(df_l1c)
    print(f"Aligned by record_idx: {n_records} test records")
    print("Paired bootstrap over records (L2A − L1C):")
    print()

    # For per-scene bootstrap: one row per record with scene_id and L1C/L2A columns
    df_merge = df_l1c[["record_idx", "scene_id"]].copy()
    for k in range(4):
        df_merge[f"l1c_c{k}"] = df_l1c[f"iou_c{k}"].values
        df_merge[f"l2a_c{k}"] = df_l2a[f"iou_c{k}"].values
    df_merge["miou_l1c"] = df_l1c["miou"].values
    df_merge["miou_l2a"] = df_l2a["miou"].values

    results = {}
    for k in range(4):
        mean_d, low, high, n_valid = bootstrap_paired_ci(
            df_l1c[f"iou_c{k}"].values, df_l2a[f"iou_c{k}"].values, B=args.B, seed=args.seed
        )
        results[f"class_{k}"] = {"mean_delta": mean_d, "ci_low": low, "ci_high": high, "n_valid": n_valid}
        label = f"Class {k} ({CLASS_LABELS.get(k, '')})".strip()
        print(f"  {label}: Δ={mean_d:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}]  (n_valid={n_valid})")

    mean_d, low, high, n_valid = bootstrap_paired_ci(
        df_l1c["miou"].values, df_l2a["miou"].values, B=args.B, seed=args.seed
    )
    results["mean_iou"] = {"mean_delta": mean_d, "ci_low": low, "ci_high": high, "n_valid": n_valid}
    print(f"  mIoU: Δ={mean_d:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}]  (n_valid={n_valid})")

    if args.per_scene:
        print()
        print("Paired bootstrap over scenes (optional):")
        results_scene = {}
        for k in range(4):
            mean_d, low, high, n_valid = bootstrap_paired_per_scene(
                df_merge, f"l1c_c{k}", f"l2a_c{k}", B=args.B, seed=args.seed
            )
            results_scene[f"class_{k}"] = {"mean_delta": mean_d, "ci_low": low, "ci_high": high, "n_valid": n_valid}
            label = f"Class {k} ({CLASS_LABELS.get(k, '')})".strip()
            print(f"  {label}: Δ={mean_d:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}]")
        mean_d, low, high, n_valid = bootstrap_paired_per_scene(
            df_merge, "miou_l1c", "miou_l2a", B=args.B, seed=args.seed
        )
        results_scene["mean_iou"] = {"mean_delta": mean_d, "ci_low": low, "ci_high": high, "n_valid": n_valid}
        print(f"  mIoU: Δ={mean_d:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}]")
        results["per_scene"] = results_scene

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_records": n_records, "B": args.B, "seed": args.seed, "results": results}, f, indent=2)
    print()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
