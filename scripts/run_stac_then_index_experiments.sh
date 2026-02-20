#!/usr/bin/env bash
# GPU 4: L1C STAC → L1C index (sequential)
# GPU 5: L2A STAC → L2A index (sequential)
# But GPU 4 and GPU 5 run in parallel
# Split 0.85/0.05/0.1 for both. Results: src/results/stac/, src/results/index/, outputs/metrics/stac|index/, outputs/stats/paired_bootstrap_ci_stac|index.json
#
# Run in background (survives SSH disconnect; use screen/tmux on a server for best resilience):
#   cd /path/to/project
#   nohup bash scripts/run_stac_then_index_experiments.sh > run_experiments.log 2>&1 &
#   tail -f run_experiments.log

set -e
cd "$(dirname "$0")/.."

# Find Python - prefer conda env if available
if command -v conda &> /dev/null; then
    # Try to use conda's python
    PYTHON_CMD=$(conda run -n satimage-env which python 2>/dev/null || which python)
elif [ -f "$HOME/anaconda3/envs/satimage-env/bin/python" ]; then
    PYTHON_CMD="$HOME/anaconda3/envs/satimage-env/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    PYTHON_CMD="python"
fi

EPOCHS=50
SPLIT="0.85,0.05,0.1"
SEED=42

echo "=== Starting parallel training ==="
echo "GPU 4: L1C STAC → L1C index (sequential)"
echo "GPU 5: L2A STAC → L2A index (sequential)"
echo ""

# GPU 4 process: L1C STAC then L1C index
(
    echo "[GPU 4] Starting L1C STAC-grouped training..."
    stdbuf -oL -eL $PYTHON_CMD -m src.main --dataset cloudsen12_l1c --epochs $EPOCHS --gpu 4 --split-ratio $SPLIT --seed $SEED --run-name stac
    L1C_STAC_EXIT=$?
    if [ $L1C_STAC_EXIT -ne 0 ]; then
        echo "[GPU 4] ERROR: L1C STAC training failed (exit=$L1C_STAC_EXIT)"
        exit $L1C_STAC_EXIT
    fi
    echo "[GPU 4] L1C STAC completed. Starting L1C index training..."
    stdbuf -oL -eL $PYTHON_CMD -m src.main --dataset cloudsen12_l1c --epochs $EPOCHS --gpu 4 --split-ratio $SPLIT --no-scene-split --seed $SEED --run-name index
    L1C_INDEX_EXIT=$?
    if [ $L1C_INDEX_EXIT -ne 0 ]; then
        echo "[GPU 4] ERROR: L1C index training failed (exit=$L1C_INDEX_EXIT)"
        exit $L1C_INDEX_EXIT
    fi
    echo "[GPU 4] All L1C trainings completed."
) > gpu4_training.log 2>&1 &
GPU4_PID=$!

# GPU 5 process: L2A STAC then L2A index
(
    echo "[GPU 5] Starting L2A STAC-grouped training..."
    stdbuf -oL -eL $PYTHON_CMD -m src.main --dataset cloudsen12_l2a --epochs $EPOCHS --gpu 5 --split-ratio $SPLIT --seed $SEED --run-name stac
    L2A_STAC_EXIT=$?
    if [ $L2A_STAC_EXIT -ne 0 ]; then
        echo "[GPU 5] ERROR: L2A STAC training failed (exit=$L2A_STAC_EXIT)"
        exit $L2A_STAC_EXIT
    fi
    echo "[GPU 5] L2A STAC completed. Starting L2A index training..."
    stdbuf -oL -eL $PYTHON_CMD -m src.main --dataset cloudsen12_l2a --epochs $EPOCHS --gpu 5 --split-ratio $SPLIT --no-scene-split --seed $SEED --run-name index
    L2A_INDEX_EXIT=$?
    if [ $L2A_INDEX_EXIT -ne 0 ]; then
        echo "[GPU 5] ERROR: L2A index training failed (exit=$L2A_INDEX_EXIT)"
        exit $L2A_INDEX_EXIT
    fi
    echo "[GPU 5] All L2A trainings completed."
) > gpu5_training.log 2>&1 &
GPU5_PID=$!

echo "GPU 4 PID: $GPU4_PID"
echo "GPU 5 PID: $GPU5_PID"
echo "Waiting for both GPUs to complete..."

wait $GPU4_PID
GPU4_EXIT=$?
wait $GPU5_PID
GPU5_EXIT=$?

if [ $GPU4_EXIT -ne 0 ] || [ $GPU5_EXIT -ne 0 ]; then
    echo "ERROR: Training failed (GPU4 exit=$GPU4_EXIT, GPU5 exit=$GPU5_EXIT)"
    exit 1
fi

echo ""
echo "=== All trainings completed. Running evaluation ==="
echo "STAC-grouped evaluation..."
$PYTHON_CMD scripts/compute_per_record_iou.py --model both --split-ratio $SPLIT --seed $SEED --run-name stac --gpu 4
$PYTHON_CMD scripts/bootstrap_paired_ci.py --run-name stac

echo "Index split evaluation..."
$PYTHON_CMD scripts/compute_per_record_iou.py --model both --split-ratio $SPLIT --no-scene-split --seed $SEED --run-name index --gpu 4
$PYTHON_CMD scripts/bootstrap_paired_ci.py --run-name index

echo "=== Done. Results: src/results/stac/, src/results/index/, outputs/metrics/stac/, outputs/metrics/index/, outputs/stats/paired_bootstrap_ci_stac.json, paired_bootstrap_ci_index.json ==="
