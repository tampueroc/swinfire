#!/bin/bash
# Quick test for checkpoints and image logging

echo "Testing checkpoints and image logger..."
echo "This will run 1 training batch + 1 validation batch"
echo ""

uv run python scripts/train_factorized_model.py \
  --override fast_dev_run=true \
  --trainer_config configs/wandb_trainer_config.yaml

echo ""
echo "Checking results..."
echo ""

# Check if checkpoints were created
if [ -d "checkpoints" ]; then
    echo "✓ Checkpoint directory exists"
    ls -lh checkpoints/ | tail -n +2
else
    echo "✗ No checkpoint directory found"
fi

echo ""

# Check WandB logs
if [ -d "wandb" ]; then
    echo "✓ WandB directory exists"
    echo "Check WandB UI for image logs at:"
    echo "https://wandb.ai/[your-entity]/swinfire-scripts/runs"
else
    echo "✗ No WandB directory found"
fi

