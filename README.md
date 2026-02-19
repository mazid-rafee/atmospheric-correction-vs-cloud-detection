# Atmospheric Correction Impact on Cloud Detectability

Cloud segmentation on CloudSen12: compare TOA (L1C) vs BOA (L2A) and quantify impact of atmospheric correction with paired bootstrap CIs.

## Setup

- Python 3.10+, PyTorch, tacoreader, rasterio, pandas, matplotlib.
- Data: CloudSen12+ TACOs in `data/CloudSen12+/TACOs/` (mini-cloudsen12-l1c-high-512.taco, mini-cloudsen12-l2a-high-512.taco).

## Run full pipeline (lock in numbers)

From the project root, same split/seed for both models:

```bash
# 1. Train both models (same seed/split so test set is identical)
python -m src.main --dataset cloudsen12_l1c --epochs 50 --gpu 0 --seed 42
python -m src.main --dataset cloudsen12_l2a --epochs 50 --gpu 0 --seed 42

# 2. Per-record IoU on test set (uses best-val checkpoints in src/results/)
python scripts/compute_per_record_iou.py --model both

# 3. Paired bootstrap 95% CI for L2A − L1C
python scripts/bootstrap_paired_ci.py

# 4. Optional: mean IoU by cloud-fraction buckets (needs GeoDataFrame from TACO)
python scripts/conditioned_results.py
```

**Outputs to use in the paper:**
- `outputs/metrics/per_record_iou_l1c.csv` and `per_record_iou_l2a.csv` — per-record IoU per class (iou_c0..c3) and mIoU.
- `outputs/stats/paired_bootstrap_ci.json` — paired bootstrap 95% CI for L2A − L1C (per-class and mIoU); optional `--per-scene` for scene-level bootstrap.
- `outputs/analysis/conditioned_iou_by_cloud_fraction.csv` — mean IoU and Δ by cloud-fraction bucket (if step 4 ran).

## Other scripts

- `scripts/verify_scene_split.py` — check STAC-group split (scenes, patches, no overlap).
- `scripts/conditioned_results.py` — mean IoU by cloud-fraction buckets (needs per_record_iou CSVs + GeoDataFrame).
- `scripts/phase1_spectral_contrast_analysis.py` — spectral Δ (TOA−BOA), contrast curves, heatmaps.

## Split

Train/val/test use a STAC-group split (scene_id from `dataset.to_geodataframe()`). In the mini TACO packaging this is nearly one patch per scene; split JSON is in `outputs/splits/`.
