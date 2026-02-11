"""
TinyLLM Text Generator
======================

Generate text from a trained TinyLLM model.
Simple and educational script for text generation.

This script loads a trained model and generates text based on a prompt.
It demonstrates the autoregressive generation process where the model
predicts one token at a time, conditioning on all previous tokens.

How Generation Works:
=====================

1. START with a prompt (seed text)
2. ENCODE the prompt into token indices
3. PREDICT the next token distribution using the model
4. SAMPLE from the distribution (with optional temperature/top-k)
5. APPEND the new token to the sequence
6. REPEAT steps 3-5 until:
   - Maximum tokens generated
   - End-of-sequence token produced
   - Keyboard interrupt

Generation Strategies:
=====================

- Greedy Decoding (do_sample=False):
  Always picks the most likely next token.
  Fast but can lead to repetitive, boring text.

- Temperature Sampling:
  - temperature=0.7: Conservative, coherent text
  - temperature=1.0: Balanced randomness
  - temperature>1.0: More creative, unpredictable
  
- Top-k Sampling:
  Only considers the k most likely tokens at each step.
  Prevents selecting very unlikely tokens.

Usage:
======
    python generate.py                           # Use default settings
    python generate.py --prompt "Hello world"    # Custom prompt
    python generate.py --tokens 500              # Generate 500 tokens
    python generate.py --temperature 0.9         # More creative
    python generate.py --topk 20                 # Limit to top-20 tokens
    python generate.py --no-sample               # Greedy decoding
    python generate.py --checkpoint best_model.pt  # Use specific checkpoint
"""

import os
import sys
import argparse
from pathlib import Path

import torch

from model import TinyLLM
from tokenizer import Tokenizer


def load_checkpoint(checkpoint_path):
    """
    Load a trained model checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
    
    Returns:
        Tuple of (model, tokenizer, info_dict)
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        print()
        print("Available checkpoints:")
        out_dir = Path("out")
        if out_dir.exists():
            for f in out_dir.glob("*.pt"):
                print(f"   - {f}")
        else:
            print("   No checkpoints found. Train first: python train.py")
        sys.exit(1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    config = checkpoint.get('config', {})
    tokenizer_data = checkpoint.get('tokenizer', {})
    
    # Create tokenizer
    tokenizer = Tokenizer(
        chars=tokenizer_data.get('chars'),
        stoi=tokenizer_data.get('stoi'),
        itos=tokenizer_data.get('itos')
    )
    
    # Create model
    model = TinyLLM(
        vocab_size=config.get('vocab_size', 65),
        n_layer=config.get('n_layer', 6),
        n_head=config.get('n_head', 6),
        n_embd=config.get('n_embd', 192),
        block_size=config.get('block_size', 256),
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✅ Checkpoint loaded!")
    print(f"   Iterations: {checkpoint.get('iter_num', 'N/A')}")
    print(f"   Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"   Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print()
    
    return model, tokenizer, checkpoint


def generate_text(model, tokenizer, prompt, max_new_tokens=200, 
                  temperature=0.8, top_k=None, do_sample=True, seed=None):
    """
    Generate text from the model.
    
    Args:
        model: TinyLLM model instance
        tokenizer: Tokenizer instance
        prompt: Starting text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = neutral)
        top_k: Limit to top-k tokens (None = no limit)
        do_sample: Whether to sample (True) or use argmax (False)
        seed: Optional random seed for reproducibility
    
    Returns:
        Generated text string (prompt + new tokens)
    """
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    
    # Encode prompt
    if isinstance(prompt, str):
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    else:
        context = prompt
    
    print(f"📝 Prompt: \"{prompt}\"")
    print(f"📏 Prompt length: {len(prompt)} characters")
    print()
    print("🤖 Generating...")
    print("-" * 60)
    
    # Generate with progress
    import time
    start_time = time.time()
    
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample
        )
    
    elapsed = time.time() - start_time
    
    # Decode
    full_text = tokenizer.decode(generated[0].tolist())
    new_text = full_text[len(prompt):]  # Only new tokens
    
    print(full_text)
    print("-" * 60)
    print()
    
    # Statistics
    num_tokens = len(generated[0]) - len(context[0])
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
    
    print(f"📊 Generation Statistics:")
    print(f"   Tokens generated: {num_tokens}")
    print(f"   Time elapsed: {elapsed:.2f}s")
    print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"   Temperature: {temperature}")
    print(f"   Top-k: {top_k if top_k else 'Disabled'}")
    print()
    
    return full_text, new_text


def interactive_generate():
    """
    Interactive text generation loop.
    
    Allows the user to enter different prompts and generation settings.
    """
    print("=" * 60)
    print("🎮 Interactive Text Generation")
    print("=" * 60)
    print()
    print("Commands:")
    print("  :set <param> <value>  - Set generation parameter")
    print("  :stats                - Show current settings")
    print("  :quit                 - Exit")
    print("  Ctrl+C                - Interrupt generation")
    print()
    
    # Default settings
    settings = {
        'max_tokens': 200,
        'temperature': 0.8,
        'top_k': None,
        'do_sample': True,
        'seed': None
    }
    
    while True:
        try:
            prompt = input("🤖 Prompt (or :cmd): ").strip()
            
            if prompt.startswith(':'):
                # Handle commands
                cmd = prompt[1:].lower().split()
                
                if not cmd:
                    continue
                
                if cmd[0] == 'quit' or cmd[0] == 'exit':
                    print("👋 Goodbye!")
                    break
                
                elif cmd[0] == 'set':
                    if len(cmd) < 3:
                        print("Usage: :set <param> <value>")
                        print("Params: max_tokens, temperature, top_k, seed")
                        print("Example: :set temperature 0.9")
                        continue
                    
                    param, value = cmd[1], cmd[2]
                    
                    if param == 'max_tokens':
                        settings['max_tokens'] = int(value)
                    elif param == 'temperature':
                        settings['temperature'] = float(value)
                    elif param == 'top_k':
                        settings['top_k'] = int(value) if value != 'none' else None
                    elif param == 'seed':
                        settings['seed'] = int(value) if value != 'none' else None
                    elif param == 'sample':
                        settings['do_sample'] = value.lower() in ('true', 'yes', '1')
                    else:
                        print(f"Unknown parameter: {param}")
                        continue
                    
                    print(f"✅ Setting updated: {param} = {value}")
                
                elif cmd[0] == 'stats':
                    print("\n📊 Current Settings:")
                    for k, v in settings.items():
                        print(f"   {k}: {v}")
                    print()
                
                else:
                    print(f"Unknown command: {cmd[0]}")
                
                continue
            
            if not prompt:
                continue
            
            # Generate
            _, _ = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                do_sample=settings['do_sample'],
                seed=settings['seed']
            )
        
        except KeyboardInterrupt:
            print("\n\n👋 Generation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate text with TinyLLM"
    )
    
    # Generation options
    parser.add_argument('--prompt', type=str, default="A",
                       help='Prompt to start generation')
    parser.add_argument('--tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--topk', type=int, default=None,
                       help='Limit to top-k tokens (None = no limit)')
    parser.add_argument('--no-sample', action='store_true',
                       help='Use greedy decoding (no sampling)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Checkpoint options
    parser.add_argument('--checkpoint', type=str, default='out/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    
    # Device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Running on: {device.upper()}")
    print()
    
    if args.interactive:
        interactive_generate()
    else:
        # Single generation
        generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.topk,
            do_sample=not args.no_sample,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
