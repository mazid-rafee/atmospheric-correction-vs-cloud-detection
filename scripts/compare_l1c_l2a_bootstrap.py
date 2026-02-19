#!/usr/bin/env python3
# Paired bootstrap L2A − L1C mean IoU (same test scenes). Run after training both with same split/seed.
import argparse
import os
import sys

PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src.data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader
from src.model.swin_crossattn_4w import Swin_CrossAttn_4W
from src.utils.trainer_tester import (
    evaluate_per_scene_iou,
    bootstrap_paired_difference_ci,
)
from src.utils.helpers import seed_everything, seed_worker


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap L2A - L1C")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=str, default="0.85,0.05,0.1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--B", type=int, default=1000)
    args = parser.parse_args()

    def parse_ratio(s):
        parts = [float(x.strip()) for x in s.split(",")]
        assert len(parts) == 3
        return tuple(parts)

    split_ratio = parse_ratio(args.split_ratio)
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_bands = list(range(1, 14))
    _, _, test_ds_l1c, _ = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(
        selected_bands, split_ratio=split_ratio, scene_split=True, seed=args.seed
    )
    _, _, test_ds_l2a, _ = cloudsen12_l2a_dataloader.get_cloudsen12_datasets(
        selected_bands, split_ratio=split_ratio, scene_split=True, seed=args.seed
    )

    test_loader_l1c = DataLoader(test_ds_l1c, batch_size=8, shuffle=False, num_workers=4)
    test_loader_l2a = DataLoader(test_ds_l2a, batch_size=8, shuffle=False, num_workers=4)

    model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
    results_dir = os.path.join(PROJECT_ROOT, "src", "results")
    path_l1c = os.path.join(results_dir, "ms_cloudcam_1xdeepcross_attn_cloudsen12_l1c_best_val.pth")
    path_l2a = os.path.join(results_dir, "ms_cloudcam_1xdeepcross_attn_cloudsen12_l2a_best_val.pth")

    if not os.path.isfile(path_l1c) or not os.path.isfile(path_l2a):
        print("Need both best_val checkpoints: cloudsen12_l1c and cloudsen12_l2a.")
        sys.exit(1)

    model.load_state_dict(torch.load(path_l1c))
    out_l1c = evaluate_per_scene_iou(
        model, test_loader_l1c, device, num_classes=4, desc="L1C test"
    )
    model.load_state_dict(torch.load(path_l2a))
    out_l2a = evaluate_per_scene_iou(
        model, test_loader_l2a, device, num_classes=4, desc="L2A test"
    )

    common_ids = sorted(set(out_l1c["scene_mean_iou"]) & set(out_l2a["scene_mean_iou"]))
    ious_l1c = [out_l1c["scene_mean_iou"][s] for s in common_ids]
    ious_l2a = [out_l2a["scene_mean_iou"][s] for s in common_ids]

    mean_diff, low, high = bootstrap_paired_difference_ci(
        ious_l1c, ious_l2a, common_ids, B=args.B, seed=args.seed
    )
    print(f"Paired bootstrap (L2A − L1C) mean IoU difference (test scenes, n={len(common_ids)}):")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  95% CI: [{low:.4f}, {high:.4f}] (B={args.B})")


if __name__ == "__main__":
    main()
