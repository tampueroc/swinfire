"""Find optimal threshold for target precision."""
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def find_threshold_for_precision(pred_probs, targets, target_precision=0.85):
    """
    Find threshold that achieves target precision.
    
    Args:
        pred_probs: [N] predicted probabilities
        targets: [N] binary targets
        target_precision: desired precision (0.85 for 85%)
    
    Returns:
        threshold, achieved_precision, achieved_recall
    """
    # Flatten if needed
    pred_probs = pred_probs.flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()
    
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(targets, pred_probs)
    
    # Find threshold closest to target precision
    idx = np.argmin(np.abs(precisions - target_precision))
    
    threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    precision = precisions[idx]
    recall = recalls[idx]
    
    return threshold, precision, recall

def plot_precision_recall_curve(pred_probs, targets, save_path='pr_curve.png'):
    """Plot precision-recall trade-off."""
    pred_probs = pred_probs.flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()
    
    precisions, recalls, thresholds = precision_recall_curve(targets, pred_probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Trade-off', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Mark 85% precision line
    plt.axhline(y=0.85, color='r', linestyle='--', label='Target Precision (85%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

# Example usage during validation:
if __name__ == "__main__":
    # Mock data for testing
    pred_probs = torch.rand(1000, 512, 512)
    targets = torch.randint(0, 2, (1000, 512, 512))
    
    threshold, precision, recall = find_threshold_for_precision(pred_probs, targets, 0.85)
    print(f"For 85% precision:")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    
    plot_precision_recall_curve(pred_probs, targets)
