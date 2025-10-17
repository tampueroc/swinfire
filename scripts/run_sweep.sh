#!/bin/bash
# Run WandB Sweep for Precision Optimization

# 1. Initialize sweep (run once)
echo "Initializing WandB sweep..."
sweep_id=$(wandb sweep configs/sweep_precision_config.yaml 2>&1 | grep "wandb agent" | awk '{print $NF}')

echo "Sweep ID: $sweep_id"
echo ""
echo "To run sweep agents:"
echo "  GPU 0: wandb agent $sweep_id"
echo "  GPU 1: CUDA_VISIBLE_DEVICES=1 wandb agent $sweep_id"
echo ""
echo "Or run in parallel:"
echo "  ./scripts/run_sweep_parallel.sh $sweep_id"
