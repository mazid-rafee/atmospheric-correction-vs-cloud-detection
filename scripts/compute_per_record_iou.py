#!/usr/bin/env python3
# Per-record IoU per class on test set. Writes outputs/metrics/per_record_iou_l1c.csv, per_record_iou_l2a.csv
# Columns: record_idx, scene_id, iou_c0, iou_c1, iou_c2, iou_c3, miou
# Usage: python scripts/compute_per_record_iou.py --model both  (or l1c / l2a)
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader
from src.model.swin_crossattn_4w import Swin_CrossAttn_4W
from src.utils.helpers import seed_worker

NUM_CLASSES = 4
REFLECTANCE_SCALE = 3000.0


def iou_f1_per_record(pred_flat, label_flat, num_classes=NUM_CLASSES):
    # IoU_k = TP/(TP+FP+FN) or nan; F1 from prec/rec
    pred_flat = np.asarray(pred_flat, dtype=np.int64).ravel()
    label_flat = np.asarray(label_flat, dtype=np.int64).ravel()
    mask = (label_flat >= 0) & (label_flat < num_classes)
    pred_flat = pred_flat[mask]
    label_flat = label_flat[mask]

    ious = []
    f1s = []
    for k in range(num_classes):
        pred_k = pred_flat == k
        gt_k = label_flat == k
        TP = (pred_k & gt_k).sum()
        FP = (pred_k & ~gt_k).sum()
        FN = (~pred_k & gt_k).sum()
        denom = TP + FP + FN
        iou = TP / denom if denom > 0 else np.nan
        ious.append(iou)
        prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
            f1s.append(np.nan)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid_ious) if valid_ious else np.nan
    return ious, f1s, miou


def run_per_record_iou(model, test_loader, device, num_classes=NUM_CLASSES):
    """Collect per-record IoU/F1. Returns list of dicts."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Per-record IoU"):
            imgs, labels, record_indices, scene_ids = batch[0], batch[1], batch[2], batch[3]
            imgs = imgs.to(device)
            if isinstance(model(imgs), tuple):
                preds = model(imgs)[0].argmax(1)
            else:
                preds = model(imgs).argmax(1)
            preds_np = preds.cpu().numpy()
            labels_np = labels.numpy()
            B = preds_np.shape[0]
            for b in range(B):
                record_idx = int(record_indices[b].item()) if torch.is_tensor(record_indices[b]) else int(record_indices[b])
                scene_id = scene_ids[b] if isinstance(scene_ids[b], str) else str(scene_ids[b])
                ious, f1s, miou = iou_f1_per_record(preds_np[b].ravel(), labels_np[b].ravel(), num_classes=num_classes)
                row = {
                    "record_idx": record_idx,
                    "scene_id": scene_id,
                    "iou_c0": ious[0], "iou_c1": ious[1], "iou_c2": ious[2], "iou_c3": ious[3],
                    "miou": miou,
                }
                rows.append(row)
    return rows


def write_csv(rows, out_path):
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["record_idx", "scene_id", "iou_c0", "iou_c1", "iou_c2", "iou_c3", "miou"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-record IoU for test set")
    parser.add_argument("--model", choices=["l1c", "l2a", "both"], default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=str, default="0.85,0.05,0.1")
    parser.add_argument("--no-scene-split", action="store_true", help="Use index-based split (must match training if you used --no-scene-split there)")
    parser.add_argument("--run-name", type=str, default="", help="Subdir for checkpoints and metrics (must match training --run-name).")
    args = parser.parse_args()

    def parse_ratio(s):
        p = [float(x.strip()) for x in s.split(",")]
        assert len(p) == 3
        return tuple(p)

    split_ratio = parse_ratio(args.split_ratio)
    run_name = (args.run_name or "").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_bands = list(range(1, 14))
    results_dir = os.path.join(PROJECT_ROOT, "src", "results", run_name) if run_name else os.path.join(PROJECT_ROOT, "src", "results")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics", run_name) if run_name else os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    to_run = []
    if args.model in ("l1c", "both"):
        to_run.append(("l1c", cloudsen12_l1c_dataloader, "ms_cloudcam_1xdeepcross_attn_cloudsen12_l1c_best_val.pth"))
    if args.model in ("l2a", "both"):
        to_run.append(("l2a", cloudsen12_l2a_dataloader, "ms_cloudcam_1xdeepcross_attn_cloudsen12_l2a_best_val.pth"))

    scene_split = not args.no_scene_split
    for flavor, dataloader_mod, ckpt_name in to_run:
        _, _, test_ds, _ = dataloader_mod.get_cloudsen12_datasets(
            selected_bands, split_ratio=split_ratio, scene_split=scene_split, seed=args.seed
        )
        print(f"Test set: {len(test_ds)} records (scene_split={scene_split}, split_ratio={split_ratio})")
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)
        ckpt_path = os.path.join(results_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            print(f"Skip {flavor}: checkpoint not found {ckpt_path}")
            continue
        model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(ckpt_path))
        rows = run_per_record_iou(model, test_loader, device)
        out_path = os.path.join(metrics_dir, f"per_record_iou_{flavor}.csv")
        write_csv(rows, out_path)


if __name__ == "__main__":
    main()
