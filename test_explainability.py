"""Test explainability features of FactorizedFireTransformer."""

import torch
from model.factorized_transformer import FactorizedFireTransformer
from model.explainability_utils import (
    visualize_gradient_attribution,
    create_explanation_report,
    compare_explanation_methods
)

def test_explainability_features():
    """Test all explainability features."""
    print("="*60)
    print("Testing Explainability Features")
    print("="*60)
    
    # Create model
    model = FactorizedFireTransformer(
        img_size=512,
        patch_size=32,
        in_channels=1,
        static_channels=8,
        num_classes=2,
        embed_dim=256,
        spatial_depth=2,  # Smaller for testing
        temporal_depth=2,
        num_heads=4,
        dropout=0.0
    )
    
    model.eval()
    
    # Create example inputs
    B, C, H, W, T = 1, 1, 512, 512, 4
    fire_seq = torch.randn(B, C, H, W, T)
    static_data = torch.randn(B, 8, H, W)
    wind_inputs = torch.randn(B, 2, T)
    valid_tokens = torch.ones(B, T)
    
    print("\n1. Testing standard gradient explanation...")
    results_grad = model.explain(
        fire_seq, static_data, wind_inputs, valid_tokens,
        method='gradient'
    )
    
    print(f"✓ Prediction shape: {results_grad['pred'].shape}")
    print(f"✓ Fire grad shape: {results_grad['grads']['fire'].shape}")
    print(f"✓ Static grad shape: {results_grad['grads']['static'].shape}")
    print(f"✓ Wind grad shape: {results_grad['grads']['wind'].shape}")
    print(f"✓ Fire grad mean: {results_grad['grads']['fire'].abs().mean().item():.6f}")
    print(f"✓ Wind grad mean: {results_grad['grads']['wind'].abs().mean().item():.6f}")
    
    print("\n2. Testing integrated gradients (this may take a minute)...")
    results_ig = model.explain(
        fire_seq, static_data, wind_inputs, valid_tokens,
        method='integrated_gradients'
    )
    
    print(f"✓ Integrated fire grad mean: {results_ig['grads']['fire'].abs().mean().item():.6f}")
    print(f"✓ Integrated wind grad mean: {results_ig['grads']['wind'].abs().mean().item():.6f}")
    
    print("\n3. Testing visualization utilities...")
    
    # Test gradient visualization
    print("  - Testing gradient attribution visualization...")
    visualize_gradient_attribution(
        results_grad['grads'],
        modality='wind',
        save_path='test_wind_attribution.png'
    )
    print("    ✓ Saved to test_wind_attribution.png")
    
    # Test comprehensive report
    print("  - Creating comprehensive explanation report...")
    target = torch.randint(0, 2, (B, 2, 400, 400)).float()  # Dummy target
    summary = create_explanation_report(
        results_grad,
        fire_seq,
        target,
        save_dir='./test_explanations',
        sample_name='test_sample'
    )
    
    print(f"\n4. Explanation Summary Statistics:")
    for key, value in summary.items():
        print(f"  - {key}: {value:.6f}")
    
    print("\n5. Comparing gradient methods...")
    compare_explanation_methods(
        model,
        fire_seq,
        static_data,
        wind_inputs,
        valid_tokens,
        save_path='test_method_comparison.png'
    )
    print("  ✓ Saved to test_method_comparison.png")
    
    print("\n" + "="*60)
    print("✅ All explainability tests passed!")
    print("="*60)
    print("\nGenerated files:")
    print("  - test_wind_attribution.png")
    print("  - test_method_comparison.png")
    print("  - test_explanations/test_sample_*.png")
    print("\nYou can now use model.explain() for training explanations!")

if __name__ == "__main__":
    test_explainability_features()
