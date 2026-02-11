"""
TinyLLM Training Script
=======================

A clean, educational training loop for TinyLLM.
Designed for beginners to understand and modify.

This script trains the language model to predict the next character
in a sequence. Through training, the model learns patterns in the text
such as grammar, vocabulary, and even some reasoning ability.

How Training Works:
===================

1. DATA PREPARATION:
   - Load text from data/input.txt
   - Tokenize into integer sequences
   - Split into training examples (inputs and targets)
   - Each example: input sequence → next character to predict

2. BATCHING:
   - Group examples into batches for efficient GPU usage
   - Each batch contains multiple sequences
   - Randomly sample positions from the data

3. FORWARD PASS:
   - Pass batch through model
   - Model outputs logits (unnormalized predictions)
   - Compute loss between predictions and actual targets

4. BACKWARD PASS:
   - Calculate gradients via automatic differentiation
   - Update model weights to minimize loss
   - Use optimizer (AdamW recommended)

5. MONITORING:
   - Track loss over time
   - Save checkpoints periodically
   - Generate samples to observe learning progress

Key Concepts for Beginners:
===========================

- Loss: Measures how wrong the model's predictions are (lower = better)
- Epoch: One pass through the entire training dataset
- Batch: Group of examples processed together
- Learning Rate: How much to adjust weights each step
- Overfitting: Model memorizes training data instead of learning patterns
- Generalization: Model's ability to handle new, unseen text

Usage:
======
    python train.py                    # Train with default settings
    python train.py --epochs 100       # Train for 100 epochs
    python train.py --lr 1e-3          # Use different learning rate
    python train.py --batch 16         # Use batch size 16
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import local modules
from model import TinyLLM, count_parameters
from tokenizer import Tokenizer, get_text_data


# ============================================================================
# HYPERPARAMETERS - Easy to modify for experiments
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for training.
    
    All hyperparameters are集中ed here for easy experimentation.
    """
    # Model architecture
    vocab_size: int = 65          # Character-level vocabulary size
    n_layer: int = 6              # Number of transformer blocks
    n_head: int = 6               # Attention heads per block
    n_embd: int = 192             # Embedding dimension
    block_size: int = 256         # Context length (max sequence)
    dropout: float = 0.1          # Regularization (0.0 = no dropout)
    
    # Training
    batch_size: int = 32          # Examples per batch
    learning_rate: float = 3e-4   # AdamW default
    weight_decay: float = 0.1     # L2 regularization
    beta1: float = 0.9            # Adam momentum
    beta2: float = 0.99           # Adam RMSprop
    max_grad_norm: float = 1.0    # Gradient clipping
    
    # Training schedule
    epochs: int = 50              # Number of training epochs
    warmup_steps: int = 100       # Learning rate warmup
    lr_decay_steps: int = None    # Steps before LR decay starts (None = halfway)
    
    # Logging and saving
    log_interval: int = 10        # Print loss every N batches
    eval_interval: int = 100      # Evaluate and generate samples every N batches
    checkpoint_interval: int = 500  # Save checkpoint every N batches
    sample_interval: int = 100    # Generate sample text every N batches
    
    # Output
    out_dir: str = "out"          # Output directory for checkpoints
    data_path: str = "data/input.txt"  # Training data file


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    """
    Dataset for language modeling.
    
    Takes a tokenized text and creates input/target pairs for training.
    Each example consists of:
    - Input: A sequence of block_size tokens
    - Target: The next token after the input
    
    By sliding a window across the text, we create many training examples.
    """
    
    def __init__(self, tokens, block_size):
        """
        Initialize dataset.
        
        Args:
            tokens: List of token indices
            block_size: Context length for each example
        """
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        """
        Number of examples in dataset.
        
        We can create an example starting at each position except
        the last one (which has no target).
        """
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Args:
            idx: Starting position in the token sequence
        
        Returns:
            input_ids: Block of tokens, shape (block_size,)
            target_ids: Next tokens to predict, shape (block_size,)
        """
        # Grab a chunk of block_size tokens
        chunk = self.tokens[idx:idx + self.block_size]
        
        # Input is the full chunk
        x = torch.tensor(chunk, dtype=torch.long)
        
        # Target is the next token for each position (shifted by 1)
        # y[i] = next token after x[i]
        # So we take chunk[1:] and need to add a dummy at the end
        # Actually, for language modeling, we predict the next token
        # So if input is [t0, t1, t2, ..., t{n-1}], we predict [t1, t2, ..., tn]
        # But we don't have tn, so we just use the next token in the chunk
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_batch(dataset, batch_size, device):
    """
    Get a random batch of data.
    
    Randomly selects batch_size examples from the dataset.
    
    Args:
        dataset: TextDataset instance
        batch_size: Number of examples per batch
        device: Device to put tensors on ('cpu' or 'cuda')
    
    Returns:
        x: Input tensor, shape (batch_size, block_size)
        y: Target tensor, shape (batch_size, block_size-1)
    """
    # Random indices into the dataset
    ix = torch.randint(0, len(dataset), (batch_size,))
    
    # Collect inputs and targets
    x = torch.zeros((batch_size, dataset.block_size), dtype=torch.long)
    y = torch.zeros((batch_size, dataset.block_size - 1), dtype=torch.long)
    
    for i, idx in enumerate(ix):
        x[i], y[i] = dataset[idx]
    
    # Move to device
    x = x.to(device)
    y = y.to(device)
    
    return x, y


def estimate_loss(model, dataloader, device, eval_iters=20):
    """
    Estimate loss on a dataset.
    
    Averages loss over multiple random batches for a more stable estimate.
    
    Args:
        model: TinyLLM model
        dataloader: DataLoader for evaluation data
        device: Device to run on
        eval_iters: Number of batches to average
    
    Returns:
        Average loss value
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    
    with torch.no_grad():
        for k in range(eval_iters):
            try:
                x, y = next(iter(dataloader))
            except StopIteration:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            logits, loss = model(x, y)
            losses[k] = loss.item()
    
    model.train()
    return losses.mean()


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="A", 
                    max_new_tokens=200, temperature=0.8, top_k=None):
    """
    Generate a sample from the model.
    
    This helps visualize what the model has learned.
    
    Args:
        model: Trained TinyLLM model
        tokenizer: Tokenizer instance
        device: Device to run on
        prompt: Starting text
        max_new_tokens: How many tokens to generate
        temperature: Randomness (higher = more creative)
        top_k: Limit to top-k most likely tokens
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    if isinstance(prompt, str):
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        context = prompt.to(device)
    
    # Generate
    generated = model.generate(
        context, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode
    text = tokenizer.decode(generated[0].tolist())
    
    model.train()
    return text


def train(config=TrainingConfig()):
    """
    Main training function.
    
    Sets up the model, data, optimizer, and runs the training loop.
    
    Args:
        config: TrainingConfig instance with all hyperparameters
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Check GPU memory
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  CUDA detected: {mem:.1f} GB GPU memory")
    else:
        print(f"🖥️  Using CPU for training")
    
    print()
    print("=" * 60)
    print("🚀 TinyLLM Training")
    print("=" * 60)
    print()
    print(f"📅 Training started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("📖 Loading training data...")
    
    if not Path(config.data_path).exists():
        print(f"❌ Error: Training data not found at '{config.data_path}'")
        print("   Please create this file with your training text.")
        print()
        print("   Example:")
        print("   $ mkdir -p data")
        print("   $ echo 'Your text here' > data/input.txt")
        sys.exit(1)
    
    # Load and tokenize text
    text = get_text_data(config.data_path)
    print(f"   Loaded {len(text):,} characters from {config.data_path}")
    
    tokenizer = Tokenizer()
    tokenizer.fit(text)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Encode entire text
    tokens = tokenizer.encode(text)
    print(f"   Total tokens: {len(tokens):,}")
    print()
    
    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    
    print("🏗️  Creating model...")
    
    model = TinyLLM(
        vocab_size=tokenizer.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        dropout=config.dropout
    ).to(device)
    
    print()
    count_parameters(model)
    
    # =========================================================================
    # CREATE DATASET AND DATALOADER
    # =========================================================================
    
    print("📦 Creating dataset...")
    
    dataset = TextDataset(tokens, config.block_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device == 'cuda'
    )
    
    print(f"   Training examples: {len(train_dataset):,}")
    print(f"   Validation examples: {len(val_dataset):,}")
    print(f"   Batches per epoch: {len(train_loader)}")
    print()
    
    # =========================================================================
    # OPTIMIZER
    # =========================================================================
    
    print("⚙️  Configuring optimizer...")
    
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device
    )
    
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Weight decay: {config.weight_decay}")
    print(f"   Optimizer: AdamW")
    print()
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("🔥 Starting training...")
    print("-" * 60)
    
    # Create output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Training state
    iter_num = 0
    best_val_loss = float('inf')
    train_loss_values = []
    val_loss_values = []
    
    # Get iterators
    train_iter = iter(train_loader)
    
    # Calculate total iterations
    total_iters = config.epochs * len(train_loader)
    print(f"   Total iterations: {total_iters:,}")
    print()
    
    # Main training loop
    progress_bar = tqdm(range(total_iters), desc="Training", ncols=80)
    
    for iter_num in progress_bar:
        # Update learning rate with warmup and decay
        if iter_num < config.warmup_steps:
            # Warmup: linearly increase LR
            lr = config.learning_rate * (iter_num + 1) / config.warmup_steps
        elif config.lr_decay_steps and iter_num > config.lr_decay_steps:
            # Decay: linearly decrease LR
            decay_progress = (iter_num - config.lr_decay_steps) / \
                            (total_iters - config.lr_decay_steps)
            lr = config.learning_rate * (1 - 0.1 * decay_progress)  # 10% decay
        else:
            lr = config.learning_rate
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Log loss
        current_loss = loss.item()
        train_loss_values.append(current_loss)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'lr': f'{lr:.1e}'
        })
        
        # Logging
        if iter_num % config.log_interval == 0:
            avg_train_loss = sum(train_loss_values[-config.log_interval:]) / \
                            len(train_loss_values[-config.log_interval:])
            print(f"   Iter {iter_num:5d} | Train Loss: {avg_train_loss:.4f} | "
                  f"LR: {lr:.1e}")
        
        # Evaluation and sampling
        if iter_num % config.eval_interval == 0:
            # Evaluate
            val_loss = estimate_loss(model, val_loader, device)
            val_loss_values.append(val_loss)
            
            print(f"\n   📊 Iter {iter_num:5d}")
            print(f"      Train Loss: {avg_train_loss:.4f}")
            print(f"      Val Loss:   {val_loss:.4f}")
            
            # Generate sample
            print(f"\n   🎲 Sample generation:")
            sample = generate_sample(
                model, tokenizer, device,
                prompt="A",
                max_new_tokens=150,
                temperature=0.8
            )
            # Show only first few lines
            sample_lines = sample.split('\n')[:5]
            for line in sample_lines:
                print(f"      {line}")
            if len(sample.split('\n')) > 5:
                print("      ...")
            print()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'train_loss': current_loss,
                    'val_loss': val_loss,
                    'config': {
                        'vocab_size': tokenizer.vocab_size,
                        'n_layer': config.n_layer,
                        'n_head': config.n_head,
                        'n_embd': config.n_embd,
                        'block_size': config.block_size,
                    },
                    'tokenizer': {
                        'chars': tokenizer.chars,
                        'stoi': tokenizer.stoi,
                        'itos': tokenizer.itos
                    }
                }
                torch.save(checkpoint, out_dir / 'best_model.pt')
                print(f"   💾 Saved best model (val_loss={val_loss:.4f})")
            
            print()
        
        # Save checkpoint
        if iter_num % config.checkpoint_interval == 0 and iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'config': {
                    'vocab_size': tokenizer.vocab_size,
                    'n_layer': config.n_layer,
                    'n_head': config.n_head,
                    'n_embd': config.n_embd,
                    'block_size': config.block_size,
                },
                'tokenizer': {
                    'chars': tokenizer.chars,
                    'stoi': tokenizer.stoi,
                    'itos': tokenizer.itos
                }
            }
            torch.save(checkpoint, out_dir / f'checkpoint_{iter_num}.pt')
            print(f"   💾 Saved checkpoint at iter {iter_num}")
    
    # =========================================================================
    # FINALIZE
    # =========================================================================
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print()
    print(f"   Final train loss: {train_loss_values[-1]:.4f}")
    print(f"   Best val loss:    {best_val_loss:.4f}")
    print(f"   Total iterations: {iter_num + 1}")
    print()
    print("📁 Output files:")
    print(f"   - {out_dir / 'best_model.pt'}")
    print(f"   - {out_dir / 'checkpoint_*.pt'}")
    print()
    print("🚀 Next steps:")
    print("   - python generate.py  # Generate text with trained model")
    print("   - python chat.py      # Chat with the model")
    print()
    
    return model, tokenizer


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Train TinyLLM language model"
    )
    
    # Training options
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data file')
    parser.add_argument('--out', type=str, default=None,
                       help='Output directory')
    
    # Model options
    parser.add_argument('--layers', type=int, default=None,
                       help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=None,
                       help='Number of attention heads')
    parser.add_argument('--embd', type=int, default=None,
                       help='Embedding dimension')
    parser.add_argument('--context', type=int, default=None,
                       help='Context length')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    
    # Override with command-line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.batch:
        config.batch_size = args.batch
    if args.lr:
        config.learning_rate = args.lr
    if args.data:
        config.data_path = args.data
    if args.out:
        config.out_dir = args.out
    if args.layers:
        config.n_layer = args.layers
    if args.heads:
        config.n_head = args.heads
    if args.embd:
        config.n_embd = args.embd
    if args.context:
        config.block_size = args.context
    
    # Print configuration
    print("⚙️  Configuration:")
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Layers: {config.n_layer}")
    print(f"   Heads: {config.n_head}")
    print(f"   Embedding dim: {config.n_embd}")
    print(f"   Context length: {config.block_size}")
    print()
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
