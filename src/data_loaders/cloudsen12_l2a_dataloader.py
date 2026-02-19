import os
import torch
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset
import tacoreader

REFLECTANCE_SCALE = 3000.0


class Cloudsen12l2aDataloader(Dataset):
    # input_mode: l1c | l2a | l1c_l2a | l1c_l2a_delta (same as L1C loader)
    def __init__(self, indices, selected_bands, idx_to_scene_id=None, input_mode="l2a"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        taco_l2a = os.path.normpath(os.path.join(base_dir, "..", "..", "data", "CloudSen12+", "TACOs", "mini-cloudsen12-l2a-high-512.taco"))
        taco_l1c = os.path.normpath(os.path.join(base_dir, "..", "..", "data", "CloudSen12+", "TACOs", "mini-cloudsen12-l1c-high-512.taco"))

        self.dataset = tacoreader.load(taco_l2a)
        self.indices = indices
        self.selected_bands = selected_bands
        self.idx_to_scene_id = idx_to_scene_id
        self.input_mode = input_mode
        self.dataset_l1c = None
        if input_mode in ("l1c", "l1c_l2a", "l1c_l2a_delta") and os.path.isfile(taco_l1c):
            self.dataset_l1c = tacoreader.load(taco_l1c)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record_idx = self.indices[idx]
        record = self.dataset.read(record_idx)
        s2_l2a_path = record.read(0)
        s2_label_path = record.read(1)

        with rio.open(s2_l2a_path) as src, rio.open(s2_label_path) as dst:
            boa = src.read(indexes=self.selected_bands).astype(np.float32) / REFLECTANCE_SCALE
            label = dst.read(1).astype(np.uint8)
        label = torch.from_numpy(label).long()

        if self.input_mode == "l2a":
            img = torch.from_numpy(boa).float()
        elif self.input_mode == "l1c":
            if self.dataset_l1c is None:
                raise FileNotFoundError("L1C TACO required for input_mode='l1c'")
            rec_l1c = self.dataset_l1c.read(record_idx)
            toa_path = rec_l1c.read(0)
            with rio.open(toa_path) as src:
                toa = src.read(indexes=self.selected_bands).astype(np.float32) / REFLECTANCE_SCALE
            img = torch.from_numpy(toa).float()
        elif self.input_mode == "l1c_l2a":
            if self.dataset_l1c is None:
                raise FileNotFoundError("L1C TACO required for input_mode='l1c_l2a'")
            rec_l1c = self.dataset_l1c.read(record_idx)
            toa_path = rec_l1c.read(0)
            with rio.open(toa_path) as src:
                toa = src.read(indexes=self.selected_bands).astype(np.float32) / REFLECTANCE_SCALE
            img = torch.from_numpy(np.concatenate([toa, boa], axis=0)).float()
        elif self.input_mode == "l1c_l2a_delta":
            if self.dataset_l1c is None:
                raise FileNotFoundError("L1C TACO required for input_mode='l1c_l2a_delta'")
            rec_l1c = self.dataset_l1c.read(record_idx)
            toa_path = rec_l1c.read(0)
            with rio.open(toa_path) as src:
                toa = src.read(indexes=self.selected_bands).astype(np.float32) / REFLECTANCE_SCALE
            delta = toa - boa
            img = torch.from_numpy(np.concatenate([toa, boa, delta], axis=0)).float()
        else:
            img = torch.from_numpy(boa).float()

        scene_id = self.idx_to_scene_id[record_idx] if self.idx_to_scene_id is not None else str(record_idx)
        return img, label, record_idx, scene_id


def get_cloudsen12_datasets(
    selected_bands,
    split_ratio=(0.85, 0.05, 0.1),
    scene_split=True,
    seed=42,
    input_mode="l2a",
):
    from src.data_loaders.cloudsen12_scene_split import (
        _get_taco_path,
        get_scene_split_indices,
    )

    split_summary = None
    idx_to_scene_id = None
    if scene_split:
        taco_path = _get_taco_path("l1c")
        train_indices, val_indices, test_indices, split_summary = get_scene_split_indices(
            taco_path, split_ratio=split_ratio, seed=seed, return_summary=True
        )
        idx_to_scene_id = split_summary.get("idx_to_scene_id")
    else:
        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError("split_ratio must sum to 1.0")
        total_samples = 10000
        indices = list(range(total_samples))
        train_end = int(split_ratio[0] * total_samples)
        val_end = train_end + int(split_ratio[1] * total_samples)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

    train_ds = Cloudsen12l2aDataloader(train_indices, selected_bands, idx_to_scene_id=idx_to_scene_id, input_mode=input_mode)
    val_ds = Cloudsen12l2aDataloader(val_indices, selected_bands, idx_to_scene_id=idx_to_scene_id, input_mode=input_mode)
    test_ds = Cloudsen12l2aDataloader(test_indices, selected_bands, idx_to_scene_id=idx_to_scene_id, input_mode=input_mode)

    return train_ds, val_ds, test_ds, split_summary
