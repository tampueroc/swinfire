#!/bin/bash
# Run multiple sweep agents in parallel (one per GPU)

if [ -z "$1" ]; then
    echo "Usage: ./run_sweep_parallel.sh <sweep_id>"
    echo "Example: ./run_sweep_parallel.sh username/project/sweep_id"
    exit 1
fi

SWEEP_ID=$1

# Activate environment
source .venv/bin/activate

# Run agent on GPU 0 in background
echo "Starting agent on GPU 0..."
CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEP_ID > logs/sweep_gpu0.log 2>&1 &
GPU0_PID=$!

# Run agent on GPU 1 in background  
echo "Starting agent on GPU 1..."
CUDA_VISIBLE_DEVICES=1 wandb agent $SWEEP_ID > logs/sweep_gpu1.log 2>&1 &
GPU1_PID=$!

echo "Agents running:"
echo "  GPU 0: PID $GPU0_PID (logs/sweep_gpu0.log)"
echo "  GPU 1: PID $GPU1_PID (logs/sweep_gpu1.log)"
echo ""
echo "To stop: kill $GPU0_PID $GPU1_PID"
echo "To monitor: tail -f logs/sweep_gpu0.log"

# Wait for both
wait $GPU0_PID $GPU1_PID
