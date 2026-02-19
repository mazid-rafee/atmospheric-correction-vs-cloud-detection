#!/usr/bin/env python3
# Mean IoU (L1C, L2A, Δ) by cloud-fraction buckets from GeoDataFrame. Writes conditioned_iou_by_cloud_fraction.csv
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders.cloudsen12_scene_split import _get_taco_path, get_scene_split_indices

METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "outputs", "analysis")
OUT_CSV = os.path.join(ANALYSIS_DIR, "conditioned_iou_by_cloud_fraction.csv")


def _normalize_col(gdf, *candidates):
    cols = [c for c in gdf.columns if c is not None]
    for cand in candidates:
        cn = str(cand).lower().replace(":", "_")
        for c in cols:
            if str(c).lower().replace(":", "_") == cn:
                return c
    return None


def main():
    taco_path = _get_taco_path("l1c")
    dataset = __import__("tacoreader").load(taco_path)
    if not hasattr(dataset, "to_geodataframe"):
        print("tacoreader dataset has no to_geodataframe(); skipping conditioned analysis.")
        sys.exit(0)
    gdf = dataset.to_geodataframe()

    _, _, test_indices, _ = get_scene_split_indices(
        taco_path, split_ratio=(0.85, 0.05, 0.1), seed=42, return_summary=True
    )
    test_indices = np.array(test_indices)

    # Columns for cloud fractions (flexible naming)
    cloud_col = _normalize_col(gdf, "cloud_percentage", "cloud_percent", "cloud")
    cloud_shadow_col = _normalize_col(gdf, "cloud_shadow_percentage", "cloud_shadow_percent")
    thin_col = _normalize_col(gdf, "thin_cloud_percentage", "thin_cloud_percent", "thin_cloud")

    csv_l1c = os.path.join(METRICS_DIR, "per_record_iou_l1c.csv")
    csv_l2a = os.path.join(METRICS_DIR, "per_record_iou_l2a.csv")
    if not os.path.isfile(csv_l1c) or not os.path.isfile(csv_l2a):
        print("Run compute_per_record_iou.py first (both L1C and L2A).")
        sys.exit(1)

    df_l1c = pd.read_csv(csv_l1c)
    df_l2a = pd.read_csv(csv_l2a)
    df_l1c = df_l1c[df_l1c["record_idx"].isin(test_indices)].copy()
    df_l2a = df_l2a[df_l2a["record_idx"].isin(test_indices)].copy()
    df_l1c = df_l1c.rename(columns={"miou": "miou_l1c"})
    df_l2a = df_l2a[["record_idx", "miou"]].rename(columns={"miou": "miou_l2a"})
    df = df_l1c[["record_idx", "miou_l1c"]].merge(df_l2a, on="record_idx")
    df["delta"] = df["miou_l2a"] - df["miou_l1c"]

    # Build metadata for test records (row index in gdf = record index)
    gdf_test = gdf.iloc[test_indices].copy()
    gdf_test["record_idx"] = test_indices
    buckets = []

    def add_quantile_col(gdf_t, col, name, nq=3):
        if col is None or col not in gdf_t.columns:
            return
        q = pd.qcut(gdf_t[col].rank(method="first"), q=nq, labels=["low", "med", "high"], duplicates="drop")
        gdf_t[name] = q.astype(str)

    add_quantile_col(gdf_test, cloud_col, "cloud_quantile")
    add_quantile_col(gdf_test, cloud_shadow_col, "cloud_shadow_quantile")
    add_quantile_col(gdf_test, thin_col, "thin_cloud_quantile")

    meta_cols = ["record_idx"]
    if "cloud_quantile" in gdf_test.columns:
        meta_cols.append("cloud_quantile")
    if "cloud_shadow_quantile" in gdf_test.columns:
        meta_cols.append("cloud_shadow_quantile")
    if "thin_cloud_quantile" in gdf_test.columns:
        meta_cols.append("thin_cloud_quantile")
    gdf_test = gdf_test[[c for c in meta_cols if c in gdf_test.columns]]
    df = df.merge(gdf_test, on="record_idx")

    # Aggregate by bucket(s)
    group_cols = [c for c in ["cloud_quantile", "cloud_shadow_quantile", "thin_cloud_quantile"] if c in df.columns]
    if not group_cols:
        print("No cloud fraction columns found in GeoDataFrame; saving overall means only.")
        rows = [{
            "bucket": "all",
            "n_records": len(df),
            "mean_miou_l1c": df["miou_l1c"].mean(),
            "mean_miou_l2a": df["miou_l2a"].mean(),
            "mean_delta": df["delta"].mean(),
        }]
    else:
        agg = df.groupby(group_cols, dropna=False).agg(
            n_records=("record_idx", "count"),
            mean_miou_l1c=("miou_l1c", "mean"),
            mean_miou_l2a=("miou_l2a", "mean"),
            mean_delta=("delta", "mean"),
        ).reset_index()
        agg["bucket"] = agg[group_cols].astype(str).agg("_".join, axis=1)
        rows = agg.to_dict("records")

    out_df = pd.DataFrame(rows)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
