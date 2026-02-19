import os
import argparse
import time
import torch
from torch.utils.data import DataLoader

from data_loaders import cloudsen12_l1c_dataloader, cloudsen12_l2a_dataloader, l8_biome_dataloader
from model.swin_crossattn_4w import Swin_CrossAttn_4W
from utils.helpers import seed_everything, seed_worker

def measure_inference_speed(model, dataloader, device, warmup=10, max_batches=40):
    model.eval()
    timings = []

    # warm-up
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            _ = model(images)
            if i >= warmup:
                break

    # timing
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            images = images.to(device)

            torch.cuda.synchronize()
            start = time.time()

            _ = model(images)

            torch.cuda.synchronize()
            end = time.time()

            timings.append(end - start)

    avg_batch_time = sum(timings) / len(timings)
    batch_size = next(iter(dataloader))[0].shape[0]

    latency_ms = (avg_batch_time / batch_size) * 1000
    fps = batch_size / avg_batch_time

    return latency_ms, fps


parser = argparse.ArgumentParser(description="Measure inference speed of MSCloudCAM")
parser.add_argument('--gpu', type=int, default=0, help='GPU index')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
seed_everything(seed)

results_dir = os.path.join("src", "results")


# ---------------------------------------------
#  CLOUDSEN12  L1C
# ---------------------------------------------
print("\n=== Measuring Inference Speed: CloudSEN12 L1C ===")

selected_bands = list(range(1, 14))
train_ds, val_ds, test_ds = cloudsen12_l1c_dataloader.get_cloudsen12_datasets(
    selected_bands, split_ratio=(0.85, 0.05, 0.1)
)

g = torch.Generator().manual_seed(seed)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

model_path = os.path.join(results_dir, "checkpoints/ms_cloudcam_cloudsen12_l1c.pth")
model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

lat, fps = measure_inference_speed(model, test_loader, device)
print(f"Latency: {lat:.2f} ms/image  |  Throughput: {fps:.2f} FPS")


# ---------------------------------------------
#  CLOUDSEN12  L2A
# ---------------------------------------------
print("\n=== Measuring Inference Speed: CloudSEN12 L2A ===")

selected_bands = list(range(1, 14))
train_ds, val_ds, test_ds = cloudsen12_l2a_dataloader.get_cloudsen12_datasets(
    selected_bands, split_ratio=(0.85, 0.05, 0.1)
)

test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

model_path = os.path.join(results_dir, "checkpoints/ms_cloudcam_cloudsen12_l2a.pth")
model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

lat, fps = measure_inference_speed(model, test_loader, device)
print(f"Latency: {lat:.2f} ms/image  |  Throughput: {fps:.2f} FPS")


# ---------------------------------------------
#  L8Biome
# ---------------------------------------------
print("\n=== Measuring Inference Speed: L8Biome ===")

selected_bands = list(range(1, 12))
train_ds, val_ds, test_ds = l8_biome_dataloader.get_l8biome_datasets(
    selected_bands, 512, 512, split_ratio=(0.6, 0.2, 0.2)
)

test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

model_path = os.path.join(results_dir, "checkpoints/ms_cloudcam_l8biome.pth")
model = Swin_CrossAttn_4W(in_channels=len(selected_bands), num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

lat, fps = measure_inference_speed(model, test_loader, device)
print(f"Latency: {lat:.2f} ms/image  |  Throughput: {fps:.2f} FPS")

print("\n=== DONE ===\n")
