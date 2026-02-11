"""
TinyLLM - Educational Language Model
=====================================

A clean, beginner-friendly implementation of a GPT-style transformer.
This model is designed for learning and experimentation, not production.

Architecture Overview:
- Decoder-only Transformer (GPT-2 style simplified)
- Character-level tokenization for transparency
- Configurable size (target: 2-3M parameters)
- CPU-optimized for consumer hardware

The Model Architecture:
======================

1. Head: Single self-attention head that computes attention weights
   - Creates Query, Key, Value projections from input
   - Computes attention: softmax(QK^T/sqrt(d)) * V
   - Allows each position to attend to all previous positions

2. MultiHeadAttention: Multiple attention heads in parallel
   - Splits embedding dimension across attention heads
   - Concatenates outputs from all heads
   - Different heads learn different patterns/relationships

3. FeedForward: Two-layer neural network
   - Expands to 4x embedding dimension (hidden layer)
   - Applies activation (GELU)
   - Projects back to original dimension
   - Adds residual connection

4. Block: Transformer block combining attention and feed-forward
   - LayerNorm → Attention → Residual
   - LayerNorm → FeedForward → Residual
   - This pattern repeated n_layer times

5. TinyLLM: Complete language model
   - Token embedding layer (vocab_size → n_embd)
   - Positional embedding (learned)
   - n_layer transformer blocks
   - LayerNorm final
   - Linear head to vocabulary

The Forward Pass:
=================
Input tokens → Embeddings + Positional → [Block × n_layer] → 
LayerNorm → Linear Head → Logits (for each position)

For training, we compute cross-entropy loss between predicted
logits and actual next-token targets.

The Generate Loop:
==================
Start with input context → Predict next token → Append to sequence →
Repeat until max_new_tokens reached. At each step, sample from
probability distribution (or use temperature for creativity).

Key Concepts for Beginners:
===========================
- Autoregressive: Predicts one token at a time, uses previous tokens as context
- Self-attention: Each token can "look at" all previous tokens
- Residual connections: Help gradients flow through deep networks
- LayerNorm: Stabilizes training by normalizing activations
- Temperature: Controls randomness (higher = more creative/unpredictable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Head(nn.Module):
    """
    Single Self-Attention Head
    
    This is the fundamental building block of transformer attention.
    Each head learns to attend to different aspects of the input.
    
    For a small model like TinyLLM, we use fewer heads with larger
    dimension per head to keep parameter count reasonable.
    
    Dimensions:
    - input: (B, T, C) where B=batch, T=time/context, C=channels/embedding dim
    - output: (B, T, head_size)
    """
    
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        
        # Query, Key, Value projections
        # These learn to project input into a space suitable for attention
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Lower triangular mask for causal attention
        # Ensures each token can only attend to previous tokens
        # Shape: (block_size, block_size) - 2D for correct broadcasting
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Compute self-attention for this head.
        
        Args:
            x: Input tensor of shape (B, T, C)
        
        Returns:
            attention output of shape (B, T, head_size)
        """
        B, T, C = x.shape
        
        # Compute Query, Key, Value
        # (B, T, C) -> (B, T, head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Compute attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # We scale by sqrt(head_size) to prevent extreme values
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        # Apply causal mask (set future positions to -inf before softmax)
        # This prevents attending to future tokens
        # Mask is 2D (T, T), broadcasts correctly with (B, T, T)
        mask_slice = self.mask[:T, :T]
        attn_scores = attn_scores.masked_fill(mask_slice == 0, float('-inf'))
        
        # Softmax to get attention weights (sum to 1 across each row)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        output = torch.matmul(attn_weights, v)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    Combines multiple attention heads in parallel.
    Each head attends to different patterns in the data.
    
    For TinyLLM:
    - n_head heads (e.g., 6 heads)
    - Each head has head_size = n_embd // n_head
    - Total dimension preserved through concatenation
    """
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = n_head
        self.head_size = n_embd // n_head
        
        # Linear projection to split into multiple heads
        # Input: (B, T, n_embd) -> Output: (B, T, n_head * head_size)
        self.heads = nn.ModuleList([
            Head(n_embd, self.head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        
        # Final projection to merge heads back to n_embd dimension
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through all attention heads.
        
        Args:
            x: Input tensor (B, T, n_embd)
        
        Returns:
            Concatenated output from all heads, projected back to n_embd
        """
        # Process through all heads and concatenate outputs
        # Each head outputs (B, T, head_size)
        # Concatenated: (B, T, n_head * head_size) = (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Final projection and dropout
        out = self.dropout(self.proj(out))
        
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    A simple two-layer neural network applied to each position independently.
    Expands the dimension, applies activation, then contracts back.
    
    This allows the model to "think" about each position's representation.
    
    Structure:
    - Input: (B, T, n_embd)
    - Expand: (B, T, n_embd * 4) via linear
    - Activate: GELU nonlinearity
    - Contract: (B, T, n_embd) via linear
    - Output: (B, T, n_embd)
    """
    
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # Expand to 4x dimension (standard in transformers)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),  # GELU is smoother than ReLU, often works better
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block
    
    The core building unit of the transformer. Each block contains:
    1. Multi-Head Self-Attention
    2. Feed-Forward Network
    
    With LayerNorm and Residual Connections (Skip Connections):
    
    x -> LayerNorm -> Attention -> Dropout -> Add x (residual)
         -> LayerNorm -> FeedForward -> Dropout -> Add x (residual)
    
    Residual connections help gradients flow through deep networks
    and allow the network to learn identity functions when helpful.
    """
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        # Self-attention sub-layer
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        
        # Feed-forward sub-layer
        self.ffwd = FeedForward(n_embd, dropout)
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        The residual connections (x + ...) help training:
        - Prevents vanishing gradients
        - Allows information to bypass layers if not needed
        """
        # Residual connection around self-attention
        x = x + self.sa(self.ln1(x))
        
        # Residual connection around feed-forward
        x = x + self.ffwd(self.ln2(x))
        
        return x


class TinyLLM(nn.Module):
    """
    TinyLLM: A Minimal Educational Language Model
    
    This is the main model class. It implements a decoder-only transformer
    (similar to GPT) trained to predict the next token in a sequence.
    
    Architecture:
    - Token Embedding: Maps vocabulary indices to dense vectors
    - Positional Embedding: Adds position information
    - Transformer Blocks: Process the sequence
    - Language Modeling Head: Maps to vocabulary logits
    
    Training:
    - Uses cross-entropy loss between predicted and actual next tokens
    - Autoregressive: predicts one token at a time
    
    Generation:
    - Start with context
    - Predict next token distribution
    - Sample from distribution
    - Append and repeat
    """
    
    def __init__(self, vocab_size, n_layer=6, n_head=6, n_embd=192, 
                 block_size=256, dropout=0.1):
        """
        Initialize the model.
        
        Args:
            vocab_size: Number of unique tokens in vocabulary
            n_layer: Number of transformer blocks
            n_head: Number of attention heads per block
            n_embd: Embedding dimension (hidden size)
            block_size: Maximum context length (position embeddings)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        # Store configuration for generation
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        
        # Token embedding table
        # Maps each token to a dense vector of size n_embd
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # Positional embedding (learned, not sinusoidal)
        # Each position gets a unique embedding vector
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        
        # Stack of transformer blocks
        # Each block: Attention → FeedForward with residual connections
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer normalization
        self.ln = nn.LayerNorm(n_embd)
        
        # Language model head: maps from n_embd to vocab_size
        # Output logits for each vocabulary token
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"🤖 TinyLLM initialized:")
        print(f"   Parameters: {self.param_count:,}")
        print(f"   Architecture: {n_layer} layers, {n_head} heads, {n_embd} dim")
        print(f"   Context: {block_size} tokens, Vocabulary: {vocab_size}")
    
    def _init_weights(self, module):
        """
        Initialize model weights.
        
        Following GPT-2 initialization:
        - Linear layers: scale by sqrt(2/model_dim)
        - Embedding layers: scale by sqrt(12/model_dim)
        - LayerNorm: scale=1.0
        """
        if isinstance(module, nn.Linear):
            # Linear layer initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm initialization
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices, shape (B, T)
            targets: Target token indices for loss computation, shape (B, T)
                    If None, loss is not computed (inference mode)
        
        Returns:
            If targets provided: tuple of (logits, loss)
            If targets None: logits only
        """
        B, T = idx.shape
        
        # Get token embeddings
        # (B, T) -> (B, T, n_embd)
        tok_emb = self.token_embedding(idx)
        
        # Add positional embeddings
        # (1, T, n_embd) broadcast to (B, T, n_embd)
        pos_emb = self.position_embedding[:, :T, :]
        x = tok_emb + pos_emb
        
        # Pass through all transformer blocks
        # Each block processes and refines the representations
        for block in self.blocks:
            x = block(x)
        
        # Final layer normalization
        x = self.ln(x)
        
        # Get logits for next-token prediction
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.head(x)
        
        # Compute loss if targets provided
        # Note: targets may be shorter than logits (T-1 vs T)
        # We only compute loss on the positions we have targets for
        loss = None
        if targets is not None:
            # Get the number of target positions
            target_len = targets.shape[1]
            
            # Use only the first target_len positions of logits
            # logits[:, :target_len] gives us (B, target_len, vocab_size)
            # Then reshape to (B * target_len, vocab_size)
            logits_for_loss = logits[:, :target_len, :].contiguous().view(-1, self.vocab_size)
            
            # Reshape targets to (B * target_len,)
            targets_flat = targets.contiguous().view(-1)
            
            loss = F.cross_entropy(logits_for_loss, targets_flat)
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4, 
                            betas=(0.9, 0.99), device_type='cpu'):
        """
        Configure the optimizer.
        
        Uses AdamW with weight decay for regularization.
        Weight decay is applied only to weights, not biases or LayerNorms.
        
        Args:
            weight_decay: L2 regularization strength
            learning_rate: Learning rate for optimizer
            betas: Adam beta coefficients (beta1=momentum, beta2=RMSprop)
            device_type: 'cpu', 'cuda', or 'mps'
        
        Returns:
            Configured torch.optim.AdamW optimizer
        """
        # Collect parameters that should be optimized
        # Exclude biases and LayerNorm weights from weight decay
        decay_params = []
        nodecay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                # LayerNorm weights and biases get no weight decay
                if name.endswith('.weight') and 'ln' in name.lower():
                    nodecay_params.append(param)
                elif param.ndim < 2:  # biases
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, 
                                     betas=betas)
        
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, 
                 top_k=None, do_sample=True):
        """
        Generate new tokens autoregressively.
        
        This is the key method that makes the model "speak". It takes
        a context (initial tokens) and extends it by predicting one
        token at a time.
        
        Args:
            idx: Starting context, shape (B, T) of token indices
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Controls randomness (1.0 = neutral, <1 = conservative, >1 = random)
            top_k: If set, only sample from top-k most likely tokens
            do_sample: If True, sample from distribution; if False, use argmax
        
        Returns:
            Generated sequence including original context
        """
        # idx is (B, T) where T is current context length
        # We want to extend it by max_new_tokens tokens
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            # Only use the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass to get logits
            # logits shape: (B, T, vocab_size)
            logits, _ = self(idx_cond)
            
            # Focus only on the last position
            # This is the prediction for the NEXT token
            # (B, T, vocab_size) -> (B, vocab_size)
            logits = logits[:, -1, :]
            
            # Apply temperature
            # Higher temperature = softer distribution (more random)
            # Lower temperature = sharper distribution (more confident)
            logits = logits / temperature
            
            # Optionally apply top-k sampling
            if top_k is not None:
                # Keep only top-k largest values
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits[:, v] = logits[:, v]
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Either sample or take argmax
            if do_sample:
                # Sample from the distribution
                # torch.multinomial returns indices based on probability
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding: always pick most likely token
                # Can lead to repetitive outputs
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            # Append new token to sequence
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def get_parameter_groups(self):
        """
        Return parameter groups for learning rate scheduling.
        
        Different layers may benefit from different learning rates.
        Typically, later layers and embedding layers use the base LR,
        while we might use lower LR for the final layer.
        
        Returns:
            List of (parameters, lr) tuples
        """
        # Main parameters with base learning rate
        main_params = list(self.blocks.parameters()) + \
                      list(self.token_embedding.parameters()) + \
                      list(self.position_embedding.parameters())
        
        # Final layer with potentially different LR
        head_params = list(self.ln.parameters()) + list(self.head.parameters())
        
        return [
            (main_params, 3e-4),
            (head_params, 3e-4),
        ]


def count_parameters(model):
    """
    Count and display model parameters breakdown.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total parameter count
    """
    total = 0
    print("\n📊 Parameter Breakdown:")
    print("-" * 40)
    
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        print(f"  {name:40s} {count:>10,}")
    
    print("-" * 40)
    print(f"  {'TOTAL':40s} {total:>10,}")
    print(f"  {'Size on disk (FP32)':40s} {total * 4 / (1024*1024):>10.2f} MB")
    print()
    
    return total


if __name__ == "__main__":
    # Demo: Create a model and print info
    print("=" * 60)
    print("🤖 TinyLLM - Educational Language Model")
    print("=" * 60)
    print()
    
    # Create model with default settings (~2M parameters)
    model = TinyLLM(
        vocab_size=65,      # Character-level (ASCII)
        n_layer=6,          # Number of transformer blocks
        n_head=6,           # Attention heads per block
        n_embd=192,         # Embedding dimension
        block_size=256,     # Context length
        dropout=0.1         # Regularization
    )
    
    print()
    count_parameters(model)
    
    # Test forward pass
    print("🧪 Testing forward pass...")
    B, T = 2, 32  # Batch size 2, sequence length 32
    idx = torch.randint(0, 65, (B, T))
    targets = torch.randint(0, 65, (B, T))
    
    logits, loss = model(idx, targets)
    print(f"  Input shape:  {idx.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n🎲 Testing generation...")
    context = torch.zeros((1, 1), dtype=torch.long)  # Start with <end> token (index 0)
    print(f"  Starting with token: {context.item()}")
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=50, temperature=0.8)
    print(f"  Generated {generated.shape[1]-1} tokens")
    print(f"  Full sequence length: {generated.shape[1]}")
    
    print("\n✅ Model test complete!")
    print("\nNext steps:")
    print("  1. python train.py     - Train the model")
    print("  2. python generate.py  - Generate text")
    print("  3. python chat.py      - Chat with the model")
