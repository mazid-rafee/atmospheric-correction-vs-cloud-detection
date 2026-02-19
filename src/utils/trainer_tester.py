import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def _unpack_batch(batch):
    imgs, labels = batch[0], batch[1]
    return imgs, labels


def train_one_epoch(model, loader, optimizer, device, desc="Training", ignore_index=None):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc=desc):
        imgs, labels = _unpack_batch(batch)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)

        if isinstance(outputs, tuple) and len(outputs) == 3:
            main_out, aux1, aux2 = outputs

            target_size = labels.shape[-2:]
            aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)

            loss_main = loss_fn(main_out, labels)
            loss_aux1 = loss_fn(aux1, labels)
            loss_aux2 = loss_fn(aux2, labels)

            loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2
        else:
            main_out = outputs
            loss = loss_fn(main_out, labels)

        if torch.isnan(loss):
            print("NaN loss detected!")
            print("Labels unique:", labels.unique())
            print("Loss input stats:", main_out.min().item(), main_out.max().item())
            raise ValueError("NaN loss. stopping")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(num_classes * labels[mask] + preds[mask], minlength=num_classes**2).reshape(num_classes, num_classes)


def evaluate_val(model, loader, device, desc="Validation", ignore_index=None):
    model.eval()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs, labels = _unpack_batch(batch)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            if isinstance(outputs, tuple) and len(outputs) == 3:
                main_out, aux1, aux2 = outputs

                target_size = labels.shape[-2:]
                aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=False)
                aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)

                loss_main = loss_fn(main_out, labels)
                loss_aux1 = loss_fn(aux1, labels)
                loss_aux2 = loss_fn(aux2, labels)

                loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2
            else:
                main_out = outputs
                loss = loss_fn(main_out, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_val_iou(model, loader, device, num_classes=4, desc="Validation"):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs, labels = _unpack_batch(batch)
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            if isinstance(outputs, tuple):
                main_out = outputs[0]
            else:
                main_out = outputs

            preds = main_out.argmax(1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().ravel(),
                                              labels.cpu().numpy().ravel(),
                                              num_classes)

    ious = []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom > 0 else 0.0
        ious.append(iou)

    return np.mean(ious)

def evaluate_test(model, loader, device, num_classes=4, desc="Testing"):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs, labels = _unpack_batch(batch)
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[0].argmax(1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().ravel(), labels.cpu().numpy().ravel(), num_classes)

    ious, f1s, lines = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"  Class {i}: IoU={iou:.4f}, F1={f1:.4f}\n")
    lines.append(f"  Mean IoU: {np.mean(ious):.4f}\n")
    lines.append(f"  Mean F1: {np.mean(f1s):.4f}\n")
    return lines


def evaluate_test_ext(model, loader, device, num_classes=4, ignore_index=None, desc="Testing"):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs, labels = _unpack_batch(batch)
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[0].argmax(1)

            if ignore_index is not None:
                valid_mask = labels != ignore_index
                preds = preds[valid_mask]
                labels = labels[valid_mask]

            preds_np = preds.cpu().numpy().ravel()
            labels_np = labels.cpu().numpy().ravel()

            if preds_np.size == 0:
                continue 

            conf_mat += fast_confusion_matrix(preds_np, labels_np, num_classes)
            total_correct += (preds_np == labels_np).sum()
            total_pixels += labels_np.size

    ious, f1s, accs, lines = [], [], [], []

    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        TN = conf_mat.sum() - (TP + FP + FN)

        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

        ious.append(iou)
        f1s.append(f1)
        accs.append(acc)

        lines.append(f"  Class {i}: IoU={iou:.4f}, F1={f1:.4f}, Acc={acc:.4f}\n")

    mIoU = np.mean(ious)
    mF1 = np.mean(f1s)
    mAcc = np.mean(accs)
    aAcc = total_correct / total_pixels if total_pixels else 0.0

    lines.append(f"\n  Mean IoU (mIoU): {mIoU:.4f}")
    lines.append(f"\n  Mean Dice/F1 (mDice): {mF1:.4f}")
    lines.append(f"\n  Mean Accuracy (mAcc): {mAcc:.4f}")
    lines.append(f"\n  Overall Accuracy (aAcc): {aAcc:.4f}")
    
    return lines


# ----- Per-scene IoU and bootstrap 95% CI (test set, scene-level) -----


def _iou_per_class_from_cm(conf_mat, num_classes):
    """IoU per class from confusion matrix. Returns list of length num_classes."""
    ious = []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        denom = TP + FP + FN
        ious.append(TP / denom if denom > 0 else 0.0)
    return ious


def evaluate_per_scene_iou(model, loader, device, num_classes=4, desc="Per-scene IoU"):
    # Batch: (img, label, record_idx, scene_id). Aggregate by scene_id, IoU per class per scene.
    model.eval()
    from collections import defaultdict

    # scene_id -> (preds list, labels list)
    scene_preds = defaultdict(list)
    scene_labels = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs, labels, scene_ids = batch[0], batch[1], batch[3]
            imgs, labels = imgs.to(device), labels.to(device)
            if isinstance(model(imgs), tuple):
                preds = model(imgs)[0].argmax(1)
            else:
                preds = model(imgs).argmax(1)
            preds_np = preds.cpu().numpy()   # (B, H, W)
            labels_np = labels.cpu().numpy()  # (B, H, W)
            B = preds_np.shape[0]
            for b in range(B):
                sid = scene_ids[b] if isinstance(scene_ids[b], str) else str(scene_ids[b])
                scene_preds[sid].extend(preds_np[b].ravel().tolist())
                scene_labels[sid].extend(labels_np[b].ravel().tolist())

    scene_iou = {}
    per_scene_mean_ious = []

    for sid in scene_preds:
        pred_arr = np.array(scene_preds[sid], dtype=np.int64)
        label_arr = np.array(scene_labels[sid], dtype=np.int64)
        cm = fast_confusion_matrix(pred_arr, label_arr, num_classes)
        ious = _iou_per_class_from_cm(cm, num_classes)
        present = []
        for i in range(num_classes):
            TP, FP = cm[i, i], cm[:, i].sum() - cm[i, i]
            FN = cm[i, :].sum() - cm[i, i]
            present.append((TP + FP + FN) > 0)
        if any(present):
            mean_iou = np.mean([iou for iou, p in zip(ious, present) if p])
        else:
            mean_iou = 0.0
        scene_iou[sid] = ious
        per_scene_mean_ious.append(mean_iou)

    global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for sid in scene_preds:
        pred_arr = np.array(scene_preds[sid], dtype=np.int64)
        label_arr = np.array(scene_labels[sid], dtype=np.int64)
        global_cm += fast_confusion_matrix(pred_arr, label_arr, num_classes)
    global_ious = _iou_per_class_from_cm(global_cm, num_classes)
    global_mean_iou = np.mean(global_ious)

    scene_mean_iou = {sid: np.mean(ious) for sid, ious in scene_iou.items()}
    return {
        "scene_iou": scene_iou,
        "scene_mean_iou": scene_mean_iou,
        "per_scene_mean_ious": per_scene_mean_ious,
        "global_ious": global_ious,
        "global_mean_iou": global_mean_iou,
        "scene_ids": list(scene_iou.keys()),
    }


def bootstrap_iou_ci(per_scene_mean_ious, B=1000, seed=42):
    if not per_scene_mean_ious:
        return 0.0, 0.0, 0.0
    arr = np.array(per_scene_mean_ious)
    n = len(arr)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(arr[idx]))
    means = np.array(means)
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_paired_difference_ci(
    per_scene_mean_ious_l1c, per_scene_mean_ious_l2a, scene_ids_common, B=1000, seed=42
):
    if not scene_ids_common or len(per_scene_mean_ious_l1c) != len(per_scene_mean_ious_l2a):
        return 0.0, 0.0, 0.0
    d = np.array(per_scene_mean_ious_l2a) - np.array(per_scene_mean_ious_l1c)
    n = len(d)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        diffs.append(np.mean(d[idx]))
    diffs = np.array(diffs)
    return float(np.mean(d)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate_test_scene_level(
    model, loader, device, num_classes=4, B=1000, seed=42, desc="Test (scene-level)",
):
    out = evaluate_per_scene_iou(model, loader, device, num_classes=num_classes, desc=desc)
    per_scene_ious = out["per_scene_mean_ious"]
    global_mean = out["global_mean_iou"]
    global_ious = out["global_ious"]

    mean_s, low, high = bootstrap_iou_ci(per_scene_ious, B=B, seed=seed)

    lines = [
        "[Scene-level evaluation (test)]",
        f"  Global mean IoU (patch-based): {global_mean:.4f}",
        f"  Per-scene mean IoU: {np.mean(per_scene_ious):.4f} (n_scenes={len(per_scene_ious)})",
        f"  Bootstrap 95% CI (per-scene mean IoU): [{low:.4f}, {high:.4f}] (B={B})",
        "",
    ]
    for c in range(num_classes):
        lines.append(f"  Class {c} global IoU: {global_ious[c]:.4f}")
    return lines, out
