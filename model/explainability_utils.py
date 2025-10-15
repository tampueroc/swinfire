"""
Explainability and Visualization Utilities for FactorizedFireTransformer.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def visualize_spatial_attention(
    attention_maps: List[torch.Tensor],
    timestep: int = 0,
    layer: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize spatial attention from Vision Transformer.
    
    Args:
        attention_maps: List of attention tensors [B, heads, patches, patches]
        timestep: Which timestep to visualize
        layer: Which layer to visualize (-1 for last layer)
        save_path: Path to save figure
        figsize: Figure size
    """
    if not attention_maps or len(attention_maps) == 0:
        print("No spatial attention maps available")
        return
    
    attn = attention_maps[layer]  # [B, heads, patches, patches]
    
    if attn.dim() == 4:
        B, H, P, _ = attn.shape
        # Average over heads
        attn_avg = attn[0].mean(dim=0)  # [patches, patches]
    else:
        attn_avg = attn[0]
    
    # Convert to numpy
    attn_np = attn_avg.numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Full attention matrix
    im1 = axes[0].imshow(attn_np, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Spatial Attention Matrix (Layer {layer}, Timestep {timestep})')
    axes[0].set_xlabel('Key Patches')
    axes[0].set_ylabel('Query Patches')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Attention from CLS token (or first patch) to all patches
    patch_side = int(np.sqrt(attn_np.shape[0]))
    first_patch_attn = attn_np[0, :]  # Attention from first patch
    
    if len(first_patch_attn) == patch_side ** 2:
        attn_2d = first_patch_attn.reshape(patch_side, patch_side)
        im2 = axes[1].imshow(attn_2d, cmap='hot', interpolation='nearest')
        axes[1].set_title('Spatial Attention Map (First Patch)')
        axes[1].set_xlabel('Patch X')
        axes[1].set_ylabel('Patch Y')
        plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial attention to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_temporal_attention(
    attention_maps: List[torch.Tensor],
    layer: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Visualize temporal attention from TransformerEncoder.
    
    Args:
        attention_maps: List of attention tensors [B, heads, T, T]
        layer: Which layer to visualize (-1 for last layer)
        save_path: Path to save figure
        figsize: Figure size
    """
    if not attention_maps or len(attention_maps) == 0:
        print("No temporal attention maps available")
        return
    
    attn = attention_maps[layer]  # [B, heads, T, T]
    
    if attn.dim() == 4:
        # Average over heads
        attn_avg = attn[0].mean(dim=0)  # [T, T]
    else:
        attn_avg = attn[0]
    
    # Convert to numpy
    attn_np = attn_avg.numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attn_np, cmap='Blues', aspect='auto')
    ax.set_title(f'Temporal Attention Matrix (Layer {layer})')
    ax.set_xlabel('Key Timesteps')
    ax.set_ylabel('Query Timesteps')
    
    # Add timestep labels
    T = attn_np.shape[0]
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f't={i}' for i in range(T)])
    ax.set_yticklabels([f't={i}' for i in range(T)])
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal attention to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_gradient_attribution(
    grads: Dict[str, torch.Tensor],
    modality: str = 'all',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize gradient-based attribution for different modalities.
    
    Args:
        grads: Dictionary with keys 'fire', 'static', 'wind'
        modality: 'all', 'fire', 'static', or 'wind'
        save_path: Path to save figure
        figsize: Figure size
    """
    if modality == 'all':
        modalities = ['fire', 'static', 'wind']
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        modalities = [modality]
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes = [axes]
    
    for ax, mod in zip(axes, modalities):
        if mod not in grads or grads[mod] is None:
            ax.text(0.5, 0.5, f'No {mod} gradients', ha='center', va='center')
            ax.set_title(f'{mod.capitalize()} Attribution')
            continue
        
        grad = grads[mod]
        
        if mod == 'fire':
            # Fire: [B, C, H, W, T] - show temporal importance
            grad_temporal = grad.abs().mean(dim=(1, 2, 3))  # [B, T]
            grad_np = grad_temporal[0].cpu().numpy()
            
            ax.bar(range(len(grad_np)), grad_np, color='orangered', alpha=0.7)
            ax.set_title('Fire Sequence Attribution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Attribution Magnitude')
            ax.set_xticks(range(len(grad_np)))
            ax.set_xticklabels([f't={i}' for i in range(len(grad_np))])
            ax.grid(axis='y', alpha=0.3)
        
        elif mod == 'static':
            # Static: [B, C, H, W] - show spatial heatmap
            grad_spatial = grad.abs().mean(dim=1)  # [B, H, W]
            grad_np = grad_spatial[0].cpu().numpy()
            
            im = ax.imshow(grad_np, cmap='YlOrRd', interpolation='bilinear')
            ax.set_title('Static Data Attribution')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Attribution')
        
        elif mod == 'wind':
            # Wind: [B, 2, T] - show temporal importance per component
            grad_np = grad[0].cpu().numpy()  # [2, T]
            
            x = np.arange(grad_np.shape[1])
            width = 0.35
            
            ax.bar(x - width/2, np.abs(grad_np[0]), width, label='U (East-West)', color='steelblue', alpha=0.7)
            ax.bar(x + width/2, np.abs(grad_np[1]), width, label='V (North-South)', color='forestgreen', alpha=0.7)
            
            ax.set_title('Wind Attribution')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Attribution Magnitude')
            ax.set_xticks(x)
            ax.set_xticklabels([f't={i}' for i in range(len(x))])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved gradient attribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_prediction_overlay(
    fire_input: torch.Tensor,
    prediction: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    timestep: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize fire input, prediction, and optionally target.
    
    Args:
        fire_input: [B, C, H, W, T] fire sequence
        prediction: [B, classes, H, W] prediction
        target: [B, classes, H, W] ground truth (optional)
        timestep: Which input timestep to show (-1 for last)
        save_path: Path to save figure
        figsize: Figure size
    """
    ncols = 3 if target is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    
    # Input fire at timestep
    fire_frame = fire_input[0, 0, :, :, timestep].cpu().numpy()
    im1 = axes[0].imshow(fire_frame, cmap='YlOrRd', interpolation='nearest')
    axes[0].set_title(f'Input Fire (t={timestep})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Prediction (apply sigmoid if logits)
    pred = prediction[0, 0].cpu().numpy() if prediction.dim() == 4 else prediction[0].cpu().numpy()
    pred_prob = 1 / (1 + np.exp(-pred))  # Sigmoid
    
    im2 = axes[1].imshow(pred_prob, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('Prediction (Probability)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Target (if provided)
    if target is not None:
        tgt = target[0, 0].cpu().numpy() if target.dim() == 4 else target[0].cpu().numpy()
        im3 = axes[2].imshow(tgt, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction overlay to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_explanation_report(
    explain_results: Dict,
    fire_input: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    save_dir: str = './explanations',
    sample_name: str = 'sample'
):
    """
    Create a comprehensive explanation report with multiple visualizations.
    
    Args:
        explain_results: Output from model.explain()
        fire_input: [B, C, H, W, T] fire sequence
        target: [B, classes, H, W] ground truth (optional)
        save_dir: Directory to save visualizations
        sample_name: Name prefix for saved files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating explanation report for {sample_name}...")
    
    # 1. Prediction overlay
    print("  - Generating prediction overlay...")
    visualize_prediction_overlay(
        fire_input,
        explain_results['pred'],
        target,
        save_path=f"{save_dir}/{sample_name}_prediction.png"
    )
    
    # 2. Gradient attribution
    print("  - Generating gradient attribution...")
    visualize_gradient_attribution(
        explain_results['grads'],
        modality='all',
        save_path=f"{save_dir}/{sample_name}_gradients.png"
    )
    
    # 3. Spatial attention (if available)
    if explain_results.get('spatial_attention') and len(explain_results['spatial_attention']) > 0:
        print("  - Generating spatial attention...")
        visualize_spatial_attention(
            explain_results['spatial_attention'],
            save_path=f"{save_dir}/{sample_name}_spatial_attention.png"
        )
    
    # 4. Temporal attention (if available)
    if explain_results.get('temporal_attention') and len(explain_results['temporal_attention']) > 0:
        print("  - Generating temporal attention...")
        visualize_temporal_attention(
            explain_results['temporal_attention'],
            save_path=f"{save_dir}/{sample_name}_temporal_attention.png"
        )
    
    print(f"✅ Explanation report saved to {save_dir}/")
    
    # Return summary statistics
    grads = explain_results['grads']
    summary = {
        'fire_grad_mean': grads['fire'].abs().mean().item() if grads['fire'] is not None else 0,
        'static_grad_mean': grads['static'].abs().mean().item() if grads['static'] is not None else 0,
        'wind_grad_mean': grads['wind'].abs().mean().item() if grads['wind'] is not None else 0,
        'prediction_mean': explain_results['pred'].mean().item(),
        'prediction_std': explain_results['pred'].std().item()
    }
    
    return summary


def compare_explanation_methods(
    model,
    fire_seq: torch.Tensor,
    static_data: torch.Tensor,
    wind_inputs: torch.Tensor,
    valid_tokens: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Compare gradient vs integrated gradients explanations.
    
    Args:
        model: FactorizedFireTransformer model
        fire_seq, static_data, wind_inputs, valid_tokens: Inputs
        save_path: Path to save comparison figure
    """
    print("Computing standard gradients...")
    results_grad = model.explain(fire_seq, static_data, wind_inputs, valid_tokens, method='gradient')
    
    print("Computing integrated gradients...")
    results_ig = model.explain(fire_seq, static_data, wind_inputs, valid_tokens, method='integrated_gradients')
    
    # Compare wind gradients (most interpretable)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Standard gradients - fire
    grad_fire = results_grad['grads']['fire'].abs().mean(dim=(1, 2, 3))[0].cpu().numpy()
    axes[0, 0].bar(range(len(grad_fire)), grad_fire, color='orangered', alpha=0.7)
    axes[0, 0].set_title('Standard Gradients - Fire')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Attribution')
    
    # Integrated gradients - fire
    ig_fire = results_ig['grads']['fire'].abs().mean(dim=(1, 2, 3))[0].cpu().numpy()
    axes[0, 1].bar(range(len(ig_fire)), ig_fire, color='orangered', alpha=0.7)
    axes[0, 1].set_title('Integrated Gradients - Fire')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Attribution')
    
    # Standard gradients - wind
    grad_wind = results_grad['grads']['wind'][0].cpu().numpy()
    x = np.arange(grad_wind.shape[1])
    axes[1, 0].bar(x - 0.2, np.abs(grad_wind[0]), 0.4, label='U', alpha=0.7)
    axes[1, 0].bar(x + 0.2, np.abs(grad_wind[1]), 0.4, label='V', alpha=0.7)
    axes[1, 0].set_title('Standard Gradients - Wind')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Attribution')
    axes[1, 0].legend()
    
    # Integrated gradients - wind
    ig_wind = results_ig['grads']['wind'][0].cpu().numpy()
    axes[1, 1].bar(x - 0.2, np.abs(ig_wind[0]), 0.4, label='U', alpha=0.7)
    axes[1, 1].bar(x + 0.2, np.abs(ig_wind[1]), 0.4, label='V', alpha=0.7)
    axes[1, 1].set_title('Integrated Gradients - Wind')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Attribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved method comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return results_grad, results_ig
