"""
TinyLLM Benchmark Suite
=======================

Performance and quality benchmarking for TinyLLM.
Measure inference speed, memory usage, and generation quality.

This script provides:
- Inference speed tests (tokens/second)
- Memory usage tests
- Generation quality tests
- Comparison between configurations

Usage:
======
    python benchmark.py                    # Run all benchmarks
    python benchmark.py --quick            # Quick benchmark
    python benchmark.py --model tiny       # Test small model
    python benchmark.py --compare          # Compare configurations
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path

from model import TinyLLM
from tokenizer import Tokenizer
from test_model import test_forward_pass, test_generation


class Benchmark:
    """Benchmark suite for TinyLLM."""
    
    def __init__(self, device='cpu'):
        """
        Initialize benchmark suite.
        
        Args:
            device: 'cpu', 'cuda', or 'auto'
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.results = {}
    
    def benchmark_inference_speed(self, model, test_prompts, warmup=10, iterations=100):
        """
        Measure inference speed.
        
        Args:
            model: TinyLLM model
            test_prompts: List of prompt strings
            warmup: Number of warmup iterations
            iterations: Number of timed iterations
        
        Returns:
            Dict with speed metrics
        """
        print("🚀 Benchmarking inference speed...")
        
        tokenizer = Tokenizer()
        tokenizer.fit("".join(test_prompts))
        
        # Warmup
        print(f"   Warming up ({warmup} iterations)...")
        model.eval()
        with torch.no_grad():
            for i in range(warmup):
                prompt = test_prompts[i % len(test_prompts)]
                context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
                _ = model.generate(context, max_new_tokens=10, temperature=0.8)
        
        # Timed runs
        print(f"   Running timed tests ({iterations} iterations)...")
        times = []
        tokens_generated = []
        
        with torch.no_grad():
            for i in range(iterations):
                prompt = test_prompts[i % len(test_prompts)]
                context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
                
                start = time.perf_counter()
                generated = model.generate(context, max_new_tokens=50, temperature=0.8)
                elapsed = time.perf_counter() - start
                
                times.append(elapsed)
                tokens_generated.append(generated.shape[1] - context.shape[1])
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        total_tokens = sum(tokens_generated)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        results = {
            'avg_time_ms': avg_time * 1000,
            'total_time_s': total_time,
            'total_tokens': total_tokens,
            'tokens_per_sec': tokens_per_sec,
            'times': times,
            'tokens_per_run': tokens_generated
        }
        
        print(f"   ✅ Results:")
        print(f"      Average time: {avg_time*1000:.2f} ms")
        print(f"      Tokens/sec: {tokens_per_sec:.1f}")
        print(f"      Total tokens: {total_tokens}")
        
        return results
    
    def benchmark_memory(self, model, batch_sizes=[1, 4, 16], context_sizes=[32, 64, 128]):
        """
        Measure memory usage for different configurations.
        
        Args:
            model: TinyLLM model
            batch_sizes: List of batch sizes to test
            context_sizes: List of context sizes to test
        
        Returns:
            Dict with memory metrics
        """
        print("\n💾 Benchmarking memory usage...")
        
        results = {}
        model.eval()
        
        for B in batch_sizes:
            for T in context_sizes:
                if B * T > model.block_size:
                    continue
                
                try:
                    # Allocate input
                    idx = torch.randint(0, 65, (B, T))
                    
                    # Measure peak memory
                    if self.device == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    
                    with torch.no_grad():
                        logits, loss = model(idx, idx[:, 1:])
                    
                    if self.device == 'cuda':
                        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                        key = f'batch{B}_ctx{T}'
                        results[key] = mem
                        print(f"   B={B}, T={T}: {mem:.2f} MB")
                    else:
                        # CPU: estimate based on tensor sizes
                        param_size = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
                        activation_size = B * T * model.n_embd * 4 / 1024 / 1024
                        total = param_size + activation_size
                        key = f'batch{B}_ctx{T}'
                        results[key] = total
                        print(f"   B={B}, T={T}: ~{total:.2f} MB (estimated)")
                        
                except Exception as e:
                    print(f"   B={B}, T={T}: Failed - {e}")
        
        return results
    
    def benchmark_quality(self, model, tokenizer, test_prompts):
        """
        Basic quality assessment.
        
        Args:
            model: TinyLLM model
            tokenizer: Tokenizer instance
            test_prompts: List of prompts to test
        
        Returns:
            Dict with quality metrics
        """
        print("\n📊 Benchmarking generation quality...")
        
        model.eval()
        results = {
            'prompts': [],
            'total_prompts': len(test_prompts),
            'unique_completions': 0,
            'avg_length': 0,
        }
        
        completions = []
        
        with torch.no_grad():
            for prompt in test_prompts:
                context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
                generated = model.generate(
                    context, 
                    max_new_tokens=100, 
                    temperature=0.8,
                    do_sample=True
                )
                
                full_text = tokenizer.decode(generated[0].tolist())
                completion = full_text[len(prompt):]
                
                completions.append(completion)
                results['prompts'].append({
                    'prompt': prompt,
                    'completion': completion[:100],  # Truncate for display
                    'length': len(completion)
                })
                
                print(f"   Prompt: '{prompt[:30]}...'")
                print(f"   Completion: '{completion[:50]}...' ({len(completion)} chars)")
        
        results['unique_completions'] = len(set(completions))
        results['avg_length'] = sum(len(c) for c in completions) / len(completions)
        
        print(f"\n   ✅ Quality metrics:")
        print(f"      Unique completions: {results['unique_completions']}/{results['total_prompts']}")
        print(f"      Avg completion length: {results['avg_length']:.1f} chars")
        
        return results
    
    def benchmark_configurations(self, configs):
        """
        Compare different model configurations.
        
        Args:
            configs: List of dicts with model parameters
        
        Returns:
            Dict with comparison results
        """
        print("\n🔄 Benchmarking different configurations...")
        print("-" * 60)
        
        results = {}
        
        for i, config in enumerate(configs, 1):
            print(f"\n{i}. Testing config: {config}")
            
            try:
                model = TinyLLM(**config).to(self.device)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"   Parameters: {param_count:,}")
                
                # Quick speed test
                tokenizer = Tokenizer()
                tokenizer.fit("Test prompt for benchmarking")
                context = torch.tensor([[0]], dtype=torch.long)
                
                model.eval()
                with torch.no_grad():
                    start = time.perf_counter()
                    for _ in range(50):
                        generated = model.generate(context, max_new_tokens=20)
                    elapsed = time.perf_counter() - start
                
                tokens_per_sec = (50 * 20) / elapsed
                print(f"   Speed: {tokens_per_sec:.1f} tok/s")
                
                results[str(config)] = {
                    'params': param_count,
                    'tok_per_sec': tokens_per_sec,
                    'config': config
                }
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return results
    
    def print_summary(self, results):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("📋 Benchmark Summary")
        print("=" * 60)
        
        for name, data in results.items():
            print(f"\n{name}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    if k != 'config':
                        print(f"   {k}: {v}")


def run_quick_benchmark():
    """Run a quick benchmark with default settings."""
    print("=" * 60)
    print("⚡ Quick Benchmark")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    
    # Create model
    print("\nCreating model...")
    model = TinyLLM(vocab_size=65, n_layer=6, n_head=6, n_embd=192)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Speed test
    print("\n🚀 Speed test...")
    tokenizer = Tokenizer()
    tokenizer.fit("Test prompt for benchmarking speed test")
    
    test_prompts = [
        "The quick brown fox",
        "Jumps over the lazy",
        "Dog and runs away",
    ]
    
    benchmark = Benchmark(device)
    
    speed_results = benchmark.benchmark_inference_speed(
        model, test_prompts, warmup=5, iterations=20
    )
    
    print("\n✅ Quick benchmark complete!")
    print(f"   Speed: {speed_results['tokens_per_sec']:.1f} tokens/sec")
    
    return True


def run_full_benchmark():
    """Run full benchmark suite."""
    print("=" * 60)
    print("🔬 Full Benchmark Suite")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    
    # Create model
    print("\nCreating model...")
    model = TinyLLM(vocab_size=65, n_layer=6, n_head=6, n_embd=192)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    benchmark = Benchmark(device)
    
    results = {}
    
    # Speed test
    tokenizer = Tokenizer()
    tokenizer.fit("The quick brown fox jumps over the lazy dog. " * 10)
    test_prompts = [
        "The quick brown fox",
        "Jumps over the lazy",
        "Dog and runs away",
        "Testing the model",
        "Language generation",
    ]
    
    results['speed'] = benchmark.benchmark_inference_speed(
        model, test_prompts, warmup=10, iterations=50
    )
    
    # Memory test
    results['memory'] = benchmark.benchmark_memory(
        model, batch_sizes=[1, 4, 16], context_sizes=[32, 64, 128]
    )
    
    # Quality test
    results['quality'] = benchmark.benchmark_quality(
        model, tokenizer, test_prompts[:3]
    )
    
    # Print summary
    benchmark.print_summary(results)
    
    print("\n✅ Full benchmark complete!")
    
    return results


def compare_configurations():
    """Compare different model configurations."""
    print("=" * 60)
    print("⚖️  Configuration Comparison")
    print("=" * 60)
    
    configs = [
        {'vocab_size': 65, 'n_layer': 3, 'n_head': 3, 'n_embd': 96},
        {'vocab_size': 65, 'n_layer': 6, 'n_head': 6, 'n_embd': 192},
        {'vocab_size': 65, 'n_layer': 8, 'n_head': 8, 'n_embd': 256},
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmark = Benchmark(device)
    results = benchmark.benchmark_configurations(configs)
    
    print("\n" + "=" * 60)
    print("📊 Comparison Results")
    print("=" * 60)
    
    for name, data in results.items():
        print(f"\nConfig: {data['config']}")
        print(f"   Parameters: {data['params']:,}")
        print(f"   Speed: {data['tok_per_sec']:.1f} tok/s")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark TinyLLM")
    
    parser.add_argument('--quick', action='store_true',
                       help="Run quick benchmark")
    parser.add_argument('--full', action='store_true',
                       help="Run full benchmark suite")
    parser.add_argument('--compare', action='store_true',
                       help="Compare different configurations")
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help="Device to use")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations()
    elif args.full:
        run_full_benchmark()
    elif args.quick:
        run_quick_benchmark()
    else:
        # Default: run quick benchmark
        run_quick_benchmark()


if __name__ == "__main__":
    main()
