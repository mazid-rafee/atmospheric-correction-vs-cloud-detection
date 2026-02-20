import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader
from src.data_loaders.cloudsen12_scene_split import format_split_summary
from src.model.swin_crossattn_4w import Swin_CrossAttn_4W
from src.utils.trainer_tester import (
    train_one_epoch,
    evaluate_val,
    evaluate_test,
    evaluate_val_iou,
    evaluate_test_scene_level,
)
from src.utils.helpers import seed_everything, seed_worker, evaluate_and_log


def parse_split_ratio(s):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--split-ratio must be three numbers, e.g. 0.85,0.05,0.1")
    return tuple(parts)


def main(epochs, gpu_id, dataset_name, split_ratio, scene_split, seed, run_name=""):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    seed_everything(seed)

    results_dir = os.path.join("src", "results", run_name) if run_name else os.path.join("src", "results")
    os.makedirs(results_dir, exist_ok=True)

    model_base_name = f"ms_cloudcam_1xdeepcross_attn_{dataset_name.lower()}"
    log_path = os.path.join(results_dir, "Cross_Attn_Segmenter.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(log_path, "a") as log_file:
        log_file.write(
            f"\n--- Run: dataset={dataset_name} split_ratio={split_ratio} "
            f"scene_split={scene_split} seed={seed} run_name={run_name or 'default'} ---\n"
        )

        if dataset_name.lower() == "cloudsen12_l1c":
            selected_bands = list(range(1, 14))
            train_ds, val_ds, test_ds, split_summary = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(
                selected_bands,
                split_ratio=split_ratio,
                scene_split=scene_split,
                seed=seed,
            )
            ignore_index = None

        elif dataset_name.lower() == "cloudsen12_l2a":
            selected_bands = list(range(1, 14))
            train_ds, val_ds, test_ds, split_summary = cloudsen12_l2a_dataloader.get_cloudsen12_datasets(
                selected_bands,
                split_ratio=split_ratio,
                scene_split=scene_split,
                seed=seed,
            )
            ignore_index = None

        else:
            raise ValueError("Invalid dataset name. Use 'cloudsen12_l1c' or 'cloudsen12_l2a'.")

        if split_summary is not None:
            print(format_split_summary(split_summary))
            log_file.write(format_split_summary(split_summary) + "\n")
            log_file.flush()
            n_patches = split_summary["n_patches_total"]
            n_scenes = split_summary["n_scenes"]
            if n_scenes >= n_patches * 0.9:
                log_file.write(
                    "  (STAC-group split is nearly patch-wise in this packaging; "
                    "each record is the independent evaluation unit.)\n"
                )
                log_file.flush()

        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

        model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_val_iou = 0.0
        best_model_path_val = os.path.join(results_dir, f"{model_base_name}_best_val.pth")

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, ignore_index=ignore_index)
            val_iou = evaluate_val_iou(model, val_loader, device, desc="Val Validation")
            test_iou = evaluate_val_iou(model, test_loader, device, desc="Test Validation")

            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val IOU = {val_iou:.4f}, Test IOU = {test_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), best_model_path_val)
                print("Saved best val model!")

        evaluate_and_log(model, best_model_path_val, test_loader, device, log_file, f"{model_base_name}_best_val")

        model.load_state_dict(torch.load(best_model_path_val))
        lines, _ = evaluate_test_scene_level(
            model, test_loader, device, num_classes=4, B=1000, seed=seed, desc=f"Scene-level {model_base_name}_best_val"
        )
        log_file.write(f"\n{model_base_name}_best_val — scene-level:\n")
        log_file.write("\n".join(lines) + "\n")
        log_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin Cross Attention 4w")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (e.g., 0, 1, 2, 3)")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cloudsen12_l1c", "cloudsen12_l2a"],
        help="Dataset to use: cloudsen12_l1c or cloudsen12_l2a",
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="0.85,0.05,0.1",
        help="Train,val,test fractions (e.g. 0.85,0.05,0.1). Used for scene-level or index split.",
    )
    parser.add_argument(
        "--no-scene-split",
        action="store_true",
        help="Use index-based split instead of scene-level split (not recommended).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training.")
    parser.add_argument("--run-name", type=str, default="", help="Subdir under src/results/ and for metrics/stats (e.g. stac, index).")
    args = parser.parse_args()

    split_ratio = parse_split_ratio(args.split_ratio)
    main(
        epochs=args.epochs,
        gpu_id=args.gpu,
        dataset_name=args.dataset,
        split_ratio=split_ratio,
        scene_split=not args.no_scene_split,
        seed=args.seed,
        run_name=args.run_name.strip(),
    )
