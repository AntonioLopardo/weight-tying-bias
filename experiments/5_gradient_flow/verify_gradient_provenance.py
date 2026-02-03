#!/usr/bin/env python
"""
Verification script for gradient provenance tracking.

This script verifies that:
1. Training with track_embedding_gradient_provenance=True produces identical weights
   to training with track_embedding_gradient_provenance=False
2. Gradient provenance metrics are correctly captured

Usage:
    python scripts/verify_gradient_provenance.py
"""

import copy
import torch
import torch.nn.functional as F

from olmo.config import ModelConfig
from olmo.model import OLMo


def create_test_model(track_provenance: bool = False) -> OLMo:
    """Create a small test model for verification."""
    config = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=1000,
        embedding_size=1024,  # Padded for efficiency
        max_sequence_length=32,
        weight_tying=True,  # Enable weight tying
        track_embedding_gradient_provenance=track_provenance,
        rope=True,
        flash_attention=False,
        init_device="cpu",
    )
    return OLMo(config)


def run_forward_backward(model: OLMo, input_ids: torch.Tensor, seed: int = 999) -> torch.Tensor:
    """Run a forward and backward pass, return the loss."""
    model.train()
    
    # Set seed for deterministic dropout
    torch.manual_seed(seed)
    
    output = model(input_ids)
    logits = output.logits

    # Simple loss: cross-entropy with shifted targets
    targets = input_ids[:, 1:].contiguous()
    logits_for_loss = logits[:, :-1, :].contiguous()
    loss = F.cross_entropy(
        logits_for_loss.view(-1, logits_for_loss.size(-1)),
        targets.view(-1),
    )
    loss.backward()
    return loss


def test_weight_equivalence():
    """Test that training with and without provenance tracking produces identical weights."""
    print("=" * 60)
    print("Test 1: Weight Equivalence")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create model without provenance tracking
    model_without = create_test_model(track_provenance=False)

    # Create model with provenance tracking by copying state dict
    model_with = create_test_model(track_provenance=True)
    model_with.load_state_dict(model_without.state_dict())
    model_with.enable_gradient_provenance_tracking()

    # Verify initial weights are identical
    for (name1, p1), (name2, p2) in zip(
        model_without.named_parameters(), model_with.named_parameters()
    ):
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
        assert torch.allclose(p1, p2), f"Initial weights differ for {name1}"

    print("✓ Initial weights are identical")

    # Create identical input
    torch.manual_seed(123)
    input_ids = torch.randint(0, 1000, (2, 32))

    # Run forward-backward on both models
    loss_without = run_forward_backward(model_without, input_ids.clone())
    loss_with = run_forward_backward(model_with, input_ids.clone())

    # Check losses are identical
    assert torch.allclose(loss_without, loss_with, atol=1e-6), (
        f"Losses differ: {loss_without.item()} vs {loss_with.item()}"
    )
    print(f"✓ Losses are identical: {loss_without.item():.6f}")

    # Check gradients are identical
    max_grad_diff = 0.0
    for (name1, p1), (name2, p2) in zip(
        model_without.named_parameters(), model_with.named_parameters()
    ):
        if p1.grad is not None and p2.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6), (
                f"Gradients differ for {name1}, max diff: {diff}"
            )

    print(f"✓ Gradients are identical (max diff: {max_grad_diff:.2e})")

    # Simulate optimizer step
    lr = 0.01
    with torch.no_grad():
        for p1, p2 in zip(model_without.parameters(), model_with.parameters()):
            if p1.grad is not None:
                p1.sub_(lr * p1.grad)
                p2.sub_(lr * p2.grad)

    # Check weights after update are identical
    max_weight_diff = 0.0
    for (name1, p1), (name2, p2) in zip(
        model_without.named_parameters(), model_with.named_parameters()
    ):
        diff = (p1 - p2).abs().max().item()
        max_weight_diff = max(max_weight_diff, diff)
        assert torch.allclose(p1, p2, atol=1e-6), (
            f"Weights differ after update for {name1}, max diff: {diff}"
        )

    print(f"✓ Weights after update are identical (max diff: {max_weight_diff:.2e})")
    print()


def test_provenance_metrics():
    """Test that gradient provenance metrics are captured correctly."""
    print("=" * 60)
    print("Test 2: Provenance Metrics Capture")
    print("=" * 60)

    torch.manual_seed(42)
    model = create_test_model(track_provenance=True)
    model.enable_gradient_provenance_tracking()

    # Before backward, metrics should be empty
    metrics = model.get_gradient_provenance_metrics()
    assert len(metrics) == 0, "Metrics should be empty before backward pass"
    print("✓ Metrics are empty before backward pass")

    # Run forward-backward
    input_ids = torch.randint(0, 1000, (2, 32))
    run_forward_backward(model, input_ids)

    # Check that metrics are now populated
    metrics = model.get_gradient_provenance_metrics()
    print(f"✓ Captured {len(metrics)} metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}")

    # Verify expected keys
    expected_keys = [
        "embedding_grad_norm",
        "embedding_grad_mean",
        "embedding_grad_abs_mean",
        "output_proj_grad_norm",
        "output_proj_grad_mean",
        "output_proj_grad_abs_mean",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing expected metric: {key}"

    print("✓ All expected metrics are present")

    # Clear and verify
    model.clear_gradient_provenance_metrics()
    metrics = model.get_gradient_provenance_metrics()
    assert len(metrics) == 0, "Metrics should be empty after clear"
    print("✓ Metrics cleared successfully")
    print()


def test_disable_tracking():
    """Test that tracking can be disabled."""
    print("=" * 60)
    print("Test 3: Enable/Disable Tracking")
    print("=" * 60)

    torch.manual_seed(42)
    model = create_test_model(track_provenance=True)

    # Enable tracking
    model.enable_gradient_provenance_tracking()
    assert model._grad_provenance_enabled, "Tracking should be enabled"
    assert len(model._grad_provenance_hooks) > 0, "Hooks should be registered"
    print("✓ Tracking enabled with hooks registered")

    # Disable tracking
    model.disable_gradient_provenance_tracking()
    assert not model._grad_provenance_enabled, "Tracking should be disabled"
    assert len(model._grad_provenance_hooks) == 0, "Hooks should be removed"
    print("✓ Tracking disabled and hooks removed")
    print()


def main():
    print("\n" + "=" * 60)
    print("Gradient Provenance Tracking Verification")
    print("=" * 60 + "\n")

    test_weight_equivalence()
    test_provenance_metrics()
    test_disable_tracking()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

