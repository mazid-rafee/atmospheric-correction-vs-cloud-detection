#!/usr/bin/env bash
# Quick status check for STAC/index experiments
cd "$(dirname "$0")/.."

echo "=== Process check ==="
if pgrep -f "run_stac_then_index_experiments.sh" > /dev/null; then
    echo "✓ Script is running"
    ps aux | grep "run_stac_then_index_experiments.sh" | grep -v grep
else
    echo "✗ Script not running"
fi

echo ""
echo "=== GPU usage (GPUs 4 and 5) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader | grep -E "^[45]," || echo "Check nvidia-smi manually"

echo ""
echo "=== STAC experiment progress ==="
if [ -f "src/results/stac/ms_cloudcam_1xdeepcross_attn_cloudsen12_l1c_best_val.pth" ]; then
    echo "✓ L1C checkpoint exists"
else
    echo "✗ L1C checkpoint missing"
fi
if [ -f "src/results/stac/ms_cloudcam_1xdeepcross_attn_cloudsen12_l2a_best_val.pth" ]; then
    echo "✓ L2A checkpoint exists"
else
    echo "✗ L2A checkpoint missing"
fi
if [ -f "outputs/metrics/stac/per_record_iou_l1c.csv" ]; then
    echo "✓ Per-record CSVs exist"
else
    echo "✗ Per-record CSVs missing"
fi
if [ -f "outputs/stats/paired_bootstrap_ci_stac.json" ]; then
    echo "✓ Bootstrap done"
else
    echo "✗ Bootstrap pending"
fi

echo ""
echo "=== Index experiment progress ==="
if [ -f "src/results/index/ms_cloudcam_1xdeepcross_attn_cloudsen12_l1c_best_val.pth" ]; then
    echo "✓ L1C checkpoint exists"
else
    echo "✗ L1C checkpoint missing"
fi
if [ -f "src/results/index/ms_cloudcam_1xdeepcross_attn_cloudsen12_l2a_best_val.pth" ]; then
    echo "✓ L2A checkpoint exists"
else
    echo "✗ L2A checkpoint missing"
fi
if [ -f "outputs/metrics/index/per_record_iou_l1c.csv" ]; then
    echo "✓ Per-record CSVs exist"
else
    echo "✗ Per-record CSVs missing"
fi
if [ -f "outputs/stats/paired_bootstrap_ci_index.json" ]; then
    echo "✓ Bootstrap done"
else
    echo "✗ Bootstrap pending"
fi

echo ""
echo "=== Latest log (last 10 lines) ==="
if [ -f "run_experiments.log" ]; then
    tail -10 run_experiments.log
else
    echo "Log file not found"
fi
