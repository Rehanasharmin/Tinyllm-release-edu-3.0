"""
TinyLLM Testing Utilities
=========================

Tests and validation for TinyLLM model.
Verify model architecture, generate samples, and test functionality.

This module provides:
- Architecture validation tests
- Forward pass tests
- Generation tests
- Checkpoint validation
- Tokenizer tests
"""

import os
import sys
import torch
from pathlib import Path

from model import TinyLLM, count_parameters
from tokenizer import Tokenizer


def test_model_creation():
    """Test that the model can be created with various configurations."""
    print("=" * 60)
    print("🧪 Test: Model Creation")
    print("=" * 60)
    print()
    
    # Test default configuration
    print("1. Testing default configuration...")
    try:
        model = TinyLLM(vocab_size=65)
        assert model.param_count > 1_000_000, "Model too small"
        assert model.param_count < 5_000_000, "Model too large"
        print(f"   ✅ Default model: {model.param_count:,} parameters")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test various configurations
    configs = [
        {"n_layer": 4, "n_head": 4, "n_embd": 128},
        {"n_layer": 8, "n_head": 8, "n_embd": 256},
        {"n_layer": 3, "n_head": 3, "n_embd": 96},
    ]
    
    for i, config in enumerate(configs, 2):
        print(f"{i}. Testing config: {config}...")
        try:
            model = TinyLLM(vocab_size=65, **config)
            print(f"   ✅ Parameters: {model.param_count:,}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    print()
    return True


def test_forward_pass():
    """Test that forward pass works correctly."""
    print("=" * 60)
    print("🧪 Test: Forward Pass")
    print("=" * 60)
    print()
    
    model = TinyLLM(vocab_size=65, n_layer=4, n_head=4, n_embd=128)
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 16]
    context_lengths = [8, 32, 64]
    
    for B in batch_sizes:
        for T in context_lengths:
            print(f"Testing shape (B={B}, T={T})...")
            
            # Create random input
            idx = torch.randint(0, 65, (B, T))
            targets = torch.randint(0, 65, (B, T))
            
            # Forward pass
            try:
                logits, loss = model(idx, targets)
                
                # Check output shape
                assert logits.shape == (B, T, 65), \
                    f"Wrong logits shape: {logits.shape}"
                
                # Check loss is valid
                assert loss.item() > 0, "Loss should be positive"
                assert not torch.isnan(loss), "Loss is NaN"
                
                print(f"   ✅ logits shape: {logits.shape}, loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return False
    
    print()
    return True


def test_generation():
    """Test that text generation works."""
    print("=" * 60)
    print("🧪 Test: Text Generation")
    print("=" * 60)
    print()
    
    model = TinyLLM(vocab_size=65, n_layer=4, n_head=4, n_embd=128)
    
    # Test generation with different settings
    test_cases = [
        {"max_new_tokens": 10, "temperature": 0.8, "do_sample": True},
        {"max_new_tokens": 20, "temperature": 1.0, "do_sample": True},
        {"max_new_tokens": 10, "temperature": 0.5, "do_sample": False},
    ]
    
    for i, kwargs in enumerate(test_cases, 1):
        print(f"Test {i}: {kwargs}...")
        try:
            # Start with single token
            idx = torch.zeros((1, 1), dtype=torch.long)
            
            # Generate
            generated = model.generate(idx, **kwargs)
            
            # Check output shape
            expected_len = 1 + kwargs["max_new_tokens"]
            assert generated.shape[1] == expected_len, \
                f"Wrong length: {generated.shape[1]} vs {expected_len}"
            
            print(f"   ✅ Generated {generated.shape[1]-1} tokens")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    print()
    return True


def test_tokenizer():
    """Test the tokenizer functionality."""
    print("=" * 60)
    print("🧪 Test: Tokenizer")
    print("=" * 60)
    print()
    
    # Create tokenizer
    tokenizer = Tokenizer()
    
    # Test learning vocabulary
    test_text = "Hello, World! How are you?"
    print(f"1. Learning vocabulary from: '{test_text}'")
    
    tokenizer.fit(test_text)
    print(f"   ✅ Vocabulary size: {tokenizer.vocab_size}")
    
    # Test encoding
    encoded = tokenizer.encode("Hello")
    print(f"2. Encoding 'Hello': {encoded}")
    assert len(encoded) == 5, "Should encode each character"
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"3. Decoding back: '{decoded}'")
    assert decoded == "Hello", f"Decode mismatch: '{decoded}'"
    print("   ✅ Encoding/decoding roundtrip works")
    
    # Test unknown characters
    unknown_encoded = tokenizer.encode("Hello©")
    print(f"4. Unknown char test: 'Hello©' -> {unknown_encoded}")
    assert unknown_encoded[-1] == 0, "Unknown char should map to <END>"
    print("   ✅ Unknown character handling works")
    
    # Test special tokens
    with_end = tokenizer.encode("Test", add_end_token=True)
    print(f"5. Special token test: 'Test<END>' -> {with_end}")
    assert with_end[-1] == 0, "Should append <END> token"
    print("   ✅ Special token works")
    
    print()
    return True


def test_checkpoint_creation():
    """Test that checkpoints can be created and loaded."""
    print("=" * 60)
    print("🧪 Test: Checkpoint Creation")
    print("=" * 60)
    print()
    
    model = TinyLLM(vocab_size=65, n_layer=4, n_head=4, n_embd=128)
    tokenizer = Tokenizer()
    tokenizer.fit("Test data for checkpoint")
    
    # Create mock checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': {},  # Empty for this test
        'iter_num': 100,
        'train_loss': 2.5,
        'val_loss': 2.4,
        'config': {
            'vocab_size': 65,
            'n_layer': 4,
            'n_head': 4,
            'n_embd': 128,
            'block_size': 256,
        },
        'tokenizer': {
            'chars': tokenizer.chars,
            'stoi': tokenizer.stoi,
            'itos': tokenizer.itos
        }
    }
    
    # Save checkpoint
    test_dir = Path("test_checkpoints")
    test_dir.mkdir(exist_ok=True)
    test_path = test_dir / "test_model.pt"
    
    print("1. Saving checkpoint...")
    try:
        torch.save(checkpoint, test_path)
        print(f"   ✅ Saved to {test_path}")
    except Exception as e:
        print(f"   ❌ Failed to save: {e}")
        return False
    
    # Load checkpoint
    print("2. Loading checkpoint...")
    try:
        loaded = torch.load(test_path, map_location='cpu')
        assert 'model' in loaded, "Missing model state dict"
        assert 'config' in loaded, "Missing config"
        print(f"   ✅ Loaded checkpoint from iter {loaded.get('iter_num', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return False
    
    # Create new model from checkpoint
    print("3. Creating model from checkpoint...")
    try:
        config = loaded['config']
        new_model = TinyLLM(
            vocab_size=config['vocab_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            block_size=config['block_size']
        )
        new_model.load_state_dict(loaded['model'])
        print(f"   ✅ Created model with {new_model.param_count:,} parameters")
    except Exception as e:
        print(f"   ❌ Failed to create model: {e}")
        return False
    
    # Cleanup
    print("4. Cleaning up test files...")
    import shutil
    shutil.rmtree(test_dir)
    print("   ✅ Test files removed")
    
    print()
    return True


def run_quick_validation():
    """Run a quick validation to ensure model works."""
    print("=" * 60)
    print("🔍 Quick Validation")
    print("=" * 60)
    print()
    
    # Create model
    print("Creating model...")
    model = TinyLLM(vocab_size=65)
    count_parameters(model)
    
    # Forward pass
    print("\nRunning forward pass...")
    B, T = 4, 32
    idx = torch.randint(0, 65, (B, T))
    targets = torch.randint(0, 65, (B, T))
    
    logits, loss = model(idx, targets)
    print(f"  Input shape: {idx.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Generate
    print("\nGenerating text...")
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=50, temperature=0.8)
    print(f"  Generated shape: {generated.shape}")
    
    # Tokenizer
    print("\nTesting tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.fit("Hello, World! Testing 123.")
    encoded = tokenizer.encode("Hi")
    decoded = tokenizer.decode(encoded)
    print(f"  'Hi' -> {encoded} -> '{decoded}'")
    
    print("\n✅ Quick validation complete!")
    return True


def run_all_tests():
    """Run all tests."""
    print()
    print("=" * 60)
    print("🧪 TinyLLM Test Suite")
    print("=" * 60)
    print()
    print("Running all tests...")
    print()
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Text Generation", test_generation),
        ("Tokenizer", test_tokenizer),
        ("Checkpoint", test_checkpoint_creation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print()
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name:20s} {status}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TinyLLM")
    parser.add_argument('--quick', action='store_true',
                       help="Run quick validation only")
    parser.add_argument('--model', action='store_true',
                       help="Test model creation")
    parser.add_argument('--forward', action='store_true',
                       help="Test forward pass")
    parser.add_argument('--generate', action='store_true',
                       help="Test generation")
    parser.add_argument('--tokenizer', action='store_true',
                       help="Test tokenizer")
    parser.add_argument('--checkpoint', action='store_true',
                       help="Test checkpoint")
    
    args = parser.parse_args()
    
    # Run tests based on arguments
    if args.quick:
        run_quick_validation()
    elif args.model:
        test_model_creation()
    elif args.forward:
        test_forward_pass()
    elif args.generate:
        test_generation()
    elif args.tokenizer:
        test_tokenizer()
    elif args.checkpoint:
        test_checkpoint_creation()
    else:
        run_all_tests()
