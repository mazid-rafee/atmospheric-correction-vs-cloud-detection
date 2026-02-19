import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import tacoreader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))

# Paths to TACOs (adjust if your directory layout differs)
TACO_DIR = os.path.join(PROJECT_ROOT, "data", "CloudSen12+", "TACOs")
TACO_L1C = os.path.join(TACO_DIR, "mini-cloudsen12-l1c-high-512.taco")  # TOA
TACO_L2A = os.path.join(TACO_DIR, "mini-cloudsen12-l2a-high-512.taco")  # BOA

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_BANDS = None  # or list of 1-based band indices
REFLECTANCE_SCALE = 3000.0

# Update to match your CloudSen12+ label mapping; script prints unique labels seen
LABEL_ROLES = {
    "clear": [0],
    "thin_cloud": [2],
    "thick_cloud": [1],
    "cloud_shadow": [3],
}


def build_label_to_role(label_roles):
    label_to_role = {}
    for role, labels in label_roles.items():
        for lab in labels:
            if lab in label_to_role and label_to_role[lab] != role:
                raise ValueError(
                    f"Label {lab} assigned to multiple roles: "
                    f"{label_to_role[lab]} and {role}"
                )
            label_to_role[lab] = role
    return label_to_role


def get_band_indices_and_count(example_raster_path, selected_bands):
    with rio.open(example_raster_path) as src:
        total_bands = src.count
        if selected_bands is None:
            band_indices = list(range(1, total_bands + 1))
        else:
            band_indices = list(selected_bands)
    return band_indices, total_bands


def iterate_tacos(taco_l1c_path, taco_l2a_path):
    ds_l1c = tacoreader.load(taco_l1c_path)
    ds_l2a = tacoreader.load(taco_l2a_path)

    n = min(len(ds_l1c), len(ds_l2a))
    for idx in range(n):
        rec_l1c = ds_l1c.read(idx)
        rec_l2a = ds_l2a.read(idx)

        # By convention in your loaders / script_pg:
        # record.read(0) -> Sentinel-2 image path
        # record.read(1) -> label path (for the L1C TACO)
        s2_l1c_path = rec_l1c.read(0)
        s2_l2a_path = rec_l2a.read(0)
        label_path = rec_l1c.read(1)

        yield idx, s2_l1c_path, s2_l2a_path, label_path


def accumulate_statistics():
    # Per-class and per-role TOA, BOA, Δ; thin-cloud Δ histograms
    label_to_role = build_label_to_role(LABEL_ROLES)

    # Will be populated after the first sample
    band_indices = None
    num_bands = None

    # Per-class accumulators
    class_sum_toa = defaultdict(lambda: None)  # label -> (B,)
    class_sum_boa = defaultdict(lambda: None)  # label -> (B,)
    class_sum_delta = defaultdict(lambda: None)  # label -> (B,)
    class_count = defaultdict(int)  # label -> number of pixels

    # Per-role accumulators (e.g. clear, thin_cloud, thick_cloud, etc.)
    role_sum_toa = defaultdict(lambda: None)  # role -> (B,)
    role_sum_boa = defaultdict(lambda: None)  # role -> (B,)
    role_count = defaultdict(int)  # role -> number of pixels

    # Δ histograms for thin cloud (per band) using fixed bins
    num_hist_bins = 100
    hist_range = (-1.0, 1.0)  # reflectance difference range (scaled 0-1, so diff in [-1, 1])
    bin_edges = None
    thin_hist_counts = None  # shape (B, num_hist_bins)

    all_labels_seen = set()

    for idx, s2_l1c_path, s2_l2a_path, label_path in iterate_tacos(TACO_L1C, TACO_L2A):
        with rio.open(s2_l1c_path) as src_l1c, rio.open(s2_l2a_path) as src_l2a, rio.open(
            label_path
        ) as src_lbl:
            if band_indices is None:
                band_indices, num_bands = get_band_indices_and_count(
                    s2_l1c_path, SELECTED_BANDS
                )

                # Initialize histogram structures once
                thin_hist_counts = np.zeros((len(band_indices), num_hist_bins), dtype=np.int64)
                bin_edges = np.linspace(hist_range[0], hist_range[1], num_hist_bins + 1)

            img_toa = (
                src_l1c.read(indexes=band_indices).astype(np.float32) / REFLECTANCE_SCALE
            )
            img_boa = (
                src_l2a.read(indexes=band_indices).astype(np.float32) / REFLECTANCE_SCALE
            )
            labels = src_lbl.read(1).astype(np.int32)

        # img_* shape: (B, H, W)
        # labels shape: (H, W)
        delta = img_toa - img_boa

        B, H, W = img_toa.shape
        img_toa_flat = img_toa.reshape(B, -1)
        img_boa_flat = img_boa.reshape(B, -1)
        delta_flat = delta.reshape(B, -1)
        labels_flat = labels.reshape(-1)

        unique_labels = np.unique(labels_flat)
        all_labels_seen.update(unique_labels.tolist())

        # ---- Per-class accumulation ----
        for lab in unique_labels:
            mask = labels_flat == lab
            if not np.any(mask):
                continue

            # (B, N_lab_pixels)
            toa_vals = img_toa_flat[:, mask]
            boa_vals = img_boa_flat[:, mask]
            delta_vals = delta_flat[:, mask]

            class_pix_count = mask.sum()

            if class_sum_toa[lab] is None:
                class_sum_toa[lab] = toa_vals.sum(axis=1)
                class_sum_boa[lab] = boa_vals.sum(axis=1)
                class_sum_delta[lab] = delta_vals.sum(axis=1)
            else:
                class_sum_toa[lab] += toa_vals.sum(axis=1)
                class_sum_boa[lab] += boa_vals.sum(axis=1)
                class_sum_delta[lab] += delta_vals.sum(axis=1)

            class_count[lab] += class_pix_count

        # ---- Per-role accumulation (cloud vs background etc.) ----
        for lab in unique_labels:
            role = label_to_role.get(lab, None)
            if role is None:
                continue

            mask = labels_flat == lab
            if not np.any(mask):
                continue

            toa_vals = img_toa_flat[:, mask]
            boa_vals = img_boa_flat[:, mask]
            role_pix_count = mask.sum()

            if role_sum_toa[role] is None:
                role_sum_toa[role] = toa_vals.sum(axis=1)
                role_sum_boa[role] = boa_vals.sum(axis=1)
            else:
                role_sum_toa[role] += toa_vals.sum(axis=1)
                role_sum_boa[role] += boa_vals.sum(axis=1)

            role_count[role] += role_pix_count

        # ---- Thin cloud Δ histograms ----
        thin_labels = LABEL_ROLES.get("thin_cloud", [])
        if thin_labels and thin_hist_counts is not None:
            thin_mask = np.isin(labels_flat, thin_labels)
            if np.any(thin_mask):
                delta_thin = delta_flat[:, thin_mask]  # (B, N_thin)
                for bi in range(len(band_indices)):
                    band_vals = delta_thin[bi, :]
                    h, _ = np.histogram(band_vals, bins=bin_edges, range=hist_range)
                    thin_hist_counts[bi] += h

    stats = {
        "band_indices": band_indices,
        "class_sum_toa": class_sum_toa,
        "class_sum_boa": class_sum_boa,
        "class_sum_delta": class_sum_delta,
        "class_count": class_count,
        "role_sum_toa": role_sum_toa,
        "role_sum_boa": role_sum_boa,
        "role_count": role_count,
        "bin_edges": bin_edges,
        "thin_hist_counts": thin_hist_counts,
        "all_labels_seen": sorted(all_labels_seen),
    }
    return stats


def compute_means_from_stats(stats):
    band_indices = stats["band_indices"]
    class_mean_toa = {}
    class_mean_boa = {}
    class_mean_delta = {}

    for lab, cnt in stats["class_count"].items():
        if cnt == 0:
            continue
        class_mean_toa[lab] = stats["class_sum_toa"][lab] / float(cnt)
        class_mean_boa[lab] = stats["class_sum_boa"][lab] / float(cnt)
        class_mean_delta[lab] = stats["class_sum_delta"][lab] / float(cnt)

    role_mean_toa = {}
    role_mean_boa = {}
    for role, cnt in stats["role_count"].items():
        if cnt == 0:
            continue
        role_mean_toa[role] = stats["role_sum_toa"][role] / float(cnt)
        role_mean_boa[role] = stats["role_sum_boa"][role] / float(cnt)

    return {
        "band_indices": band_indices,
        "class_mean_toa": class_mean_toa,
        "class_mean_boa": class_mean_boa,
        "class_mean_delta": class_mean_delta,
        "role_mean_toa": role_mean_toa,
        "role_mean_boa": role_mean_boa,
    }


def plot_delta_heatmap(means, output_dir):
    band_indices = means["band_indices"]
    class_mean_delta = means["class_mean_delta"]

    if not class_mean_delta:
        return

    classes = sorted(class_mean_delta.keys())
    matrix = np.vstack([class_mean_delta[c] for c in classes])

    vmax = np.max(np.abs(matrix))

    plt.figure(figsize=(10, 6))
    im = plt.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Δ (TOA - BOA)")
    plt.xlabel("Band index (1-based)")
    plt.ylabel("Label")
    plt.xticks(
        np.arange(len(band_indices)),
        [str(b) for b in band_indices],
        rotation=90,
    )
    plt.yticks(np.arange(len(classes)), [str(c) for c in classes])
    plt.title("Per-class band-wise Δ (TOA - BOA)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_heatmap_class_by_band.png"), dpi=200)
    plt.close()


def plot_spectral_curves(means, output_dir, label_subset=None):
    band_indices = means["band_indices"]
    class_mean_toa = means["class_mean_toa"]
    class_mean_boa = means["class_mean_boa"]

    if not class_mean_toa:
        return

    all_labels = sorted(class_mean_toa.keys())
    labels_to_plot = label_subset if label_subset is not None else all_labels

    for lab in labels_to_plot:
        if lab not in class_mean_toa:
            continue
        toa = class_mean_toa[lab]
        boa = class_mean_boa[lab]

        plt.figure(figsize=(7, 4))
        plt.plot(band_indices, toa, "-o", label="TOA (L1C)")
        plt.plot(band_indices, boa, "-o", label="BOA (L2A)")
        plt.xlabel("Band index (1-based)")
        plt.ylabel("Reflectance")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f"Spectral curves for label {lab}")
        plt.tight_layout()
        fname = os.path.join(output_dir, f"spectral_curves_label_{lab}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_contrast_curves(means, output_dir):
    # Needs LABEL_ROLES: clear + at least one cloud role
    band_indices = means["band_indices"]
    role_mean_toa = means["role_mean_toa"]
    role_mean_boa = means["role_mean_boa"]

    if "clear" not in role_mean_toa:
        return

    clear_toa = role_mean_toa["clear"]
    clear_boa = role_mean_boa["clear"]

    # Aggregate all cloud roles (if any)
    cloud_roles = [r for r in role_mean_toa.keys() if r != "clear"]
    if not cloud_roles:
        return

    # Simple average over all cloud roles (weighted by pixel count would be better,
    # but we only stored mean per-role; if you need weighted, extend the accumulators).
    cloud_toa_list = [role_mean_toa[r] for r in cloud_roles]
    cloud_boa_list = [role_mean_boa[r] for r in cloud_roles]

    cloud_toa = np.mean(np.vstack(cloud_toa_list), axis=0)
    cloud_boa = np.mean(np.vstack(cloud_boa_list), axis=0)

    C_toa = np.abs(cloud_toa - clear_toa)
    C_boa = np.abs(cloud_boa - clear_boa)
    dC = C_toa - C_boa

    # Contrast curves
    plt.figure(figsize=(7, 4))
    plt.plot(band_indices, C_toa, "-o", label="Contrast TOA (L1C)")
    plt.plot(band_indices, C_boa, "-o", label="Contrast BOA (L2A)")
    plt.xlabel("Band index (1-based)")
    plt.ylabel("Cloud–background contrast")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Cloud–background contrast (clear vs aggregated clouds)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contrast_curves_clear_vs_clouds.png"), dpi=200)
    plt.close()

    # Contrast drop
    plt.figure(figsize=(7, 4))
    plt.plot(band_indices, dC, "-o")
    plt.xlabel("Band index (1-based)")
    plt.ylabel("Contrast drop (C_TOA - C_BOA)")
    plt.grid(True, alpha=0.3)
    plt.title("Contrast drop due to atmospheric correction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contrast_drop_clear_vs_clouds.png"), dpi=200)
    plt.close()


def plot_delta_histograms_thin_cloud(stats, means, output_dir, band_subset=None):
    band_indices = means["band_indices"]
    thin_hist_counts = stats["thin_hist_counts"]
    bin_edges = stats["bin_edges"]

    if thin_hist_counts is None or bin_edges is None:
        return

    if band_subset is None:
        bands_to_plot = range(len(band_indices))
    else:
        # band_subset is a list of actual 1-based band indices;
        # convert to positions in band_indices
        index_to_pos = {b: i for i, b in enumerate(band_indices)}
        bands_to_plot = [index_to_pos[b] for b in band_subset if b in index_to_pos]

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for pos in bands_to_plot:
        b_idx = band_indices[pos]
        counts = thin_hist_counts[pos]

        plt.figure(figsize=(7, 4))
        plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], align="center")
        plt.xlabel("Δ (TOA - BOA)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.title(f"Δ histogram for thin cloud – Band {b_idx}")
        plt.tight_layout()
        fname = os.path.join(output_dir, f"delta_hist_thin_cloud_band_{b_idx}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def main():
    print("Running Phase 1 — Spectral & Contrast Analysis")
    print(f"L1C TACO: {TACO_L1C}")
    print(f"L2A TACO: {TACO_L2A}")
    print(f"Outputs will be saved to: {OUTPUT_DIR}")

    stats = accumulate_statistics()
    means = compute_means_from_stats(stats)

    print("Unique labels seen in dataset:", stats["all_labels_seen"])
    print("Configured label roles:", LABEL_ROLES)
    print("Band indices used (1-based):", means["band_indices"])

    # Δ heatmap (class × band)
    plot_delta_heatmap(means, OUTPUT_DIR)

    # Spectral curves (TOA vs BOA) for all labels
    plot_spectral_curves(means, OUTPUT_DIR)

    # Contrast curves and contrast drop (requires clear/cloud roles)
    plot_contrast_curves(means, OUTPUT_DIR)

    # Δ histograms for thin cloud (if thin_cloud labels are configured)
    plot_delta_histograms_thin_cloud(stats, means, OUTPUT_DIR)

    print("Phase 1 analysis completed.")


if __name__ == "__main__":
    main()

