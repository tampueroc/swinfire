"""Test script for FactorizedFireTransformer."""

import torch
from model.factorized_transformer import FactorizedFireTransformer

def test_forward_pass():
    """Test basic forward pass with example inputs."""
    
    # Create model
    model = FactorizedFireTransformer(
        img_size=512,
        patch_size=32,
        in_channels=1,
        static_channels=8,
        num_classes=2,
        embed_dim=256,  # Smaller for testing
        spatial_depth=2,
        temporal_depth=2,
        num_heads=4,
        dropout=0.1,
        optimizer_settings={'optimizer': 'adam', 'learning_rate': 1e-3},
        loss_fn='bce'
    )
    
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create example inputs
    B, C, H, W, T = 2, 1, 512, 512, 4
    fire_seq = torch.randn(B, C, H, W, T)
    static_data = torch.randn(B, 8, H, W)
    wind_inputs = torch.randn(B, 2, T)
    valid_tokens = torch.ones(B, T)
    
    print("\nInput shapes:")
    print(f"  fire_seq: {fire_seq.shape}")
    print(f"  static_data: {static_data.shape}")
    print(f"  wind_inputs: {wind_inputs.shape}")
    print(f"  valid_tokens: {valid_tokens.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(fire_seq, static_data, wind_inputs, valid_tokens)
    
    print(f"Output shape: {output.shape}")
    print(f"Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")
    
    # Test with variable length sequences
    print("\n\nTesting with masked timesteps...")
    valid_tokens_masked = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],  # All valid
        [1.0, 1.0, 1.0, 0.0],  # Last masked
    ])
    
    with torch.no_grad():
        output_masked = model(fire_seq, static_data, wind_inputs, valid_tokens_masked)
    
    print(f"Output shape (masked): {output_masked.shape}")
    
    # Test explain mode
    print("\n\nTesting explainability mode...")
    model.eval()
    explain_results = model.explain(
        fire_seq[:1],  # Single sample
        static_data[:1],
        wind_inputs[:1],
        valid_tokens[:1]
    )
    
    print("Explain results keys:", explain_results.keys())
    print(f"  pred shape: {explain_results['pred'].shape}")
    if explain_results['grads']['fire'] is not None:
        print(f"  fire grad shape: {explain_results['grads']['fire'].shape}")
        print(f"  fire grad mean: {explain_results['grads']['fire'].abs().mean().item():.6f}")
    if explain_results['grads']['static'] is not None:
        print(f"  static grad shape: {explain_results['grads']['static'].shape}")
        print(f"  static grad mean: {explain_results['grads']['static'].abs().mean().item():.6f}")
    if explain_results['grads']['wind'] is not None:
        print(f"  wind grad shape: {explain_results['grads']['wind'].shape}")
        print(f"  wind grad mean: {explain_results['grads']['wind'].abs().mean().item():.6f}")
    
    print("\n✅ All tests passed!")
    
    return model

def test_training_step():
    """Test training step."""
    print("\n" + "="*60)
    print("Testing training step...")
    print("="*60)
    
    model = FactorizedFireTransformer(
        img_size=512,
        patch_size=64,  # Larger patch for faster testing
        in_channels=1,
        static_channels=8,
        num_classes=2,
        embed_dim=128,  # Smaller
        spatial_depth=2,
        temporal_depth=2,
        num_heads=4,
        dropout=0.0,
        optimizer_settings={'optimizer': 'adam', 'learning_rate': 1e-3},
        loss_fn='bce'
    )
    
    # Create batch
    B, C, H, W, T = 2, 1, 512, 512, 4
    fire_seq = torch.randn(B, C, H, W, T)
    static_data = torch.randn(B, 8, H, W)
    wind_inputs = torch.randn(B, 2, T)
    
    # Target (cropped)
    isochrone_mask = torch.randint(0, 2, (B, 2, 400, 400)).float()
    valid_tokens = torch.ones(B, T)
    
    batch = (fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens)
    
    # Training step
    model.train()
    result = model.training_step(batch, 0)
    
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"Predictions shape: {result['predictions'].shape}")
    print(f"Targets shape: {result['targets'].shape}")
    
    # Backward pass
    result['loss'].backward()
    print("✅ Backward pass successful!")
    
    print("\n✅ Training step test passed!")

if __name__ == "__main__":
    print("="*60)
    print("Testing FactorizedFireTransformer")
    print("="*60)
    
    model = test_forward_pass()
    test_training_step()
    
    print("\n" + "="*60)
    print("🎉 All tests completed successfully!")
    print("="*60)
