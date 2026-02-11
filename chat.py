"""
TinyLLM Interactive Chat
========================

An interactive chat interface for TinyLLM.
Pure text generation - NO hardcoded responses.

This is the key difference from production models: the model generates
all responses based on its training. This means:
- Responses vary based on training data
- Model may say unexpected things
- Model may make mistakes
- Model truly "learns" rather than "memorizes"

How It Works:
=============

1. User enters text
2. Text is formatted as a conversation: "User: <input>\nAssistant: "
3. Model generates continuation (the response)
4. Response is extracted and displayed
5. Conversation history is maintained (up to context limit)

The model learns to:
- Respond to questions (based on training examples)
- Follow conversational patterns
- Generate relevant text
- Maintain coherence within context

If the model hasn't been trained, it will generate random characters.
This is a feature! It shows beginners that TRAINING is required.

Usage:
======
    python chat.py                      # Use default settings
    python chat.py --checkpoint best_model.pt  # Use specific checkpoint
    python chat.py --temp 0.7           # Lower temperature = more focused
    python chat.py --context 512        # Larger context window
    python chat.py --no-history         # No conversation history

Educational Notes:
==================

For beginners, this chat interface demonstrates:

1. Language Models are Pattern Matchers:
   They don't "know" things - they match patterns from training data.
   If trained on conversations, they'll generate conversation-like text.

2. Temperature Controls Creativity:
   - Low temperature (0.5): More predictable, repetitive
   - Medium temperature (0.8): Balanced
   - High temperature (1.2+): More random, sometimes nonsensical

3. Context Window is Limited:
   The model can only "remember" a fixed number of tokens.
   Old conversation is forgotten when context is full.

4. Training Data Matters:
   The model's personality comes entirely from training data.
   Train on conversations → learns to converse.
   Train on code → learns to write code.
   Train on nothing → generates random text.
"""

import os
import sys
import argparse
from pathlib import Path

import torch

from model import TinyLLM
from tokenizer import Tokenizer


class Conversation:
    """
    Manages conversation history.
    
    Maintains a rolling window of conversation context
    to provide the model with conversation history.
    """
    
    def __init__(self, system_prompt=None, max_context=256):
        """
        Initialize conversation.
        
        Args:
            system_prompt: Optional system message to set behavior
            max_context: Maximum context tokens to maintain
        """
        self.messages = []
        self.max_context = max_context
        
        if system_prompt:
            self.messages.append(('system', system_prompt))
    
    def add_message(self, role, content):
        """
        Add a message to the conversation.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message text
        """
        self.messages.append((role, content))
    
    def get_prompt(self):
        """
        Format conversation as a prompt for the model.
        
        Returns:
            Formatted prompt string
        """
        lines = []
        
        for role, content in self.messages:
            if role == 'system':
                lines.append(content)
            elif role == 'user':
                lines.append(f"User: {content}")
            elif role == 'assistant':
                lines.append(f"Assistant: {content}")
        
        return '\n'.join(lines) + '\nAssistant: '
    
    def get_response(self, generated_text):
        """
        Extract assistant response from generated text.
        
        Args:
            generated_text: Full generated text including prompt
        
        Returns:
            Assistant response string
        """
        # Find where assistant response starts
        if 'Assistant:' in generated_text:
            response = generated_text.split('Assistant:')[-1].strip()
        else:
            # Fallback: take everything after the last newline
            response = generated_text.split('\n')[-1].strip()
        
        # Clean up any continuation markers
        for marker in ['User:', 'Assistant:', 'user:', 'assistant:']:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        return response
    
    def truncate(self):
        """
        Truncate conversation if it exceeds context limit.
        
        Keeps the most recent messages while staying within limit.
        """
        # Calculate current token count (rough estimate: 4 chars per token)
        total_chars = sum(len(content) for _, content in self.messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens <= self.max_context:
            return
        
        # Truncate from the beginning
        while estimated_tokens > self.max_context and len(self.messages) > 1:
            # Remove oldest non-system message
            if self.messages[0][0] == 'system':
                # Remove second message if first is system
                self.messages.pop(1)
            else:
                self.messages.pop(0)
            
            total_chars = sum(len(content) for _, content in self.messages)
            estimated_tokens = total_chars // 4
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []


def load_model(checkpoint_path):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"📂 Loading model: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Error: Model not found: {checkpoint_path}")
        print()
        print("Please train the model first:")
        print("   python train.py")
        print()
        print("Or specify a checkpoint:")
        print("   python chat.py --checkpoint out/best_model.pt")
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
        dropout=0.0
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✅ Model loaded!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Context window: {model.block_size} tokens")
    print(f"   Vocabulary: {tokenizer.vocab_size} tokens")
    print()
    
    return model, tokenizer, checkpoint


def chat_session(model, tokenizer, args):
    """
    Run an interactive chat session.
    
    Args:
        model: TinyLLM model
        tokenizer: Tokenizer instance
        args: Command-line arguments
    """
    # Create conversation
    conversation = Conversation(
        system_prompt="You are TinyLLM, a helpful AI assistant. You respond "
                     "to user questions and comments in a friendly, concise manner.",
        max_context=args.context
    )
    
    # Display header
    print("=" * 60)
    print("💬 TinyLLM Chat - Pure Text Generation")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT: This model GENERATES text from training.")
    print("   It does NOT use hardcoded responses or rules.")
    print("   If untrained, it will generate random characters.")
    print()
    print("Commands:")
    print("  :quit, :exit, :q  - Exit chat")
    print("  :clear            - Clear conversation history")
    print("  :stats            - Show model info")
    print("  :temp <value>     - Set temperature (0.1-2.0)")
    print()
    print("-" * 60)
    print()
    
    # Check if model has been trained
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    iter_num = checkpoint.get('iter_num', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    if iter_num == 0:
        print("⚠️  WARNING: Model has not been trained!")
        print("   Running inference on random weights.")
        print("   Expected output: Random characters")
        print()
    elif iter_num < 100:
        print(f"⚠️  Model is undertrained ({iter_num} iterations).")
        print("   Loss is still high, responses may be poor.")
        print()
    else:
        print(f"✅ Model trained for {iter_num} iterations.")
        print(f"   Validation loss: {val_loss:.4f}")
        print()
    
    print("Type :help for commands, or start chatting!")
    print()
    
    # Settings
    temperature = args.temperature
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle commands
            if user_input.startswith(':'):
                command = user_input[1:].lower().split()
                
                if not command:
                    continue
                
                cmd = command[0]
                
                if cmd in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for chatting!")
                    break
                
                elif cmd == 'clear':
                    conversation.clear()
                    print("🧹 Conversation cleared!")
                    print()
                    continue
                
                elif cmd == 'stats':
                    param_count = sum(p.numel() for p in model.parameters())
                    print(f"\n📊 Model Statistics:")
                    print(f"   Parameters: {param_count:,}")
                    print(f"   Context: {args.context} tokens")
                    print(f"   Temperature: {temperature}")
                    print(f"   Messages: {len([m for m in conversation.messages if m[0] != 'system'])}")
                    print()
                    continue
                
                elif cmd == 'temp':
                    if len(command) > 1:
                        try:
                            new_temp = float(command[1])
                            if 0.1 <= new_temp <= 2.0:
                                temperature = new_temp
                                print(f"🌡️  Temperature set to {temperature}")
                            else:
                                print("❌ Temperature must be between 0.1 and 2.0")
                        except ValueError:
                            print("❌ Invalid temperature value")
                    else:
                        print(f"🌡️  Current temperature: {temperature}")
                    print()
                    continue
                
                elif cmd == 'help':
                    print("\n📖 Commands:")
                    print("  :quit       - Exit chat")
                    print("  :clear      - Clear conversation")
                    print("  :stats      - Show model info")
                    print("  :temp 0.8   - Set temperature")
                    print()
                    continue
                
                else:
                    print(f"❓ Unknown command: {cmd}")
                    print()
                    continue
            
            if not user_input:
                continue
            
            # Add user message to conversation
            conversation.add_message('user', user_input)
            
            # Truncate if needed
            conversation.truncate()
            
            # Get prompt
            prompt = conversation.get_prompt()
            
            # Generate response
            print("🤖 TinyLLM is thinking...", end='\r')
            sys.stdout.flush()
            
            with torch.no_grad():
                context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
                generated = model.generate(
                    context,
                    max_new_tokens=args.max_tokens,
                    temperature=temperature,
                    top_k=args.topk,
                    do_sample=not args.greedy
                )
            
            # Extract response
            full_text = tokenizer.decode(generated[0].tolist())
            response = conversation.get_response(full_text)
            
            # Clear "thinking" message
            print(" " * 30, end='\r')
            
            # Add assistant message to conversation
            conversation.add_message('assistant', response)
            
            # Display response
            print(f"TinyLLM: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Type :quit to exit and try again.")
            print()


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Chat with TinyLLM"
    )
    
    # Model options
    parser.add_argument('--checkpoint', type=str, default='out/best_model.pt',
                       help='Path to model checkpoint')
    
    # Generation options
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens per response')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.1-2.0)')
    parser.add_argument('--topk', type=int, default=None,
                       help='Top-k sampling (None = disabled)')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding (no sampling)')
    parser.add_argument('--context', type=int, default=200,
                       help='Context window size')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, checkpoint = load_model(args.checkpoint)
    
    # Run chat
    chat_session(model, tokenizer, args)


if __name__ == "__main__":
    main()
