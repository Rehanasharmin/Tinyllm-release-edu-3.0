"""
TinyLLM Tokenizer
=================

A simple character-level tokenizer for educational purposes.

Character-level tokenization is the simplest approach to language modeling:
- Each character in the text becomes a token
- Vocabulary is just the set of unique characters
- Easy to understand and debug
- No subword learning algorithms needed

For beginners, character-level tokenization provides maximum transparency:
- You can see exactly what the model sees
- No mysterious BPE merges or byte-pair encoding
- Easy to inspect and understand

Trade-offs:
- Larger context needed to form meaningful words
- More tokens required for the same text
- Model must learn spelling and grammar from scratch

For production models, subword tokenization (BPE, WordPiece, SentencePiece)
is typically used for better efficiency.
"""

import os
import json
from pathlib import Path


class Tokenizer:
    """
    Simple Character-Level Tokenizer
    
    This tokenizer maps between characters and integer indices.
    It preserves the order of characters and handles any text.
    
    Special Tokens:
    - <END>: End of sequence (index 0)
    - All other characters: mapped to indices 1 to vocab_size-1
    
    The tokenizer automatically learns the vocabulary from the training data.
    """
    
    def __init__(self, chars=None, stoi=None, itos=None):
        """
        Initialize tokenizer.
        
        Can be initialized in three ways:
        1. With chars: Learn vocabulary from characters
        2. With stoi/itos: Load existing vocabulary mappings
        3. Empty: Will need to be fitted with data
        
        Args:
            chars: Optional list/set of characters for vocabulary
            stoi: Optional string-to-index dictionary
            itos: Optional index-to-string dictionary
        """
        # Always have <END> as first token (index 0)
        self.special_tokens = ['<END>']
        
        if chars is not None:
            # Build vocabulary from characters
            self.chars = self.special_tokens + sorted(list(set(chars)))
            self.vocab_size = len(self.chars)
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
        elif stoi is not None and itos is not None:
            # Load existing vocabulary
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
            # Reconstruct chars list
            self.chars = [itos[i] for i in range(self.vocab_size)]
        else:
            # Empty tokenizer (needs to be fitted)
            self.chars = self.special_tokens.copy()
            self.vocab_size = len(self.chars)
            self.stoi = {'<END>': 0}
            self.itos = {0: '<END>'}
    
    def fit(self, text):
        """
        Learn vocabulary from text data.
        
        Scans through the text and collects all unique characters.
        
        Args:
            text: String of training data
        
        Returns:
            self (for method chaining)
        """
        # Get unique characters from text
        unique_chars = sorted(list(set(text)))
        
        # Build vocabulary
        self.chars = self.special_tokens + unique_chars
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        return self
    
    def encode(self, text, add_end_token=False):
        """
        Convert text to token indices.
        
        Args:
            text: Input string
            add_end_token: Whether to append <END> token at the end
        
        Returns:
            List of integer indices
        """
        ids = []
        for char in text:
            # Get index for character, default to <END> if unknown
            idx = self.stoi.get(char, 0)  # 0 is <END>
            ids.append(idx)
        
        if add_end_token:
            ids.append(0)  # <END> token
        
        return ids
    
    def decode(self, ids, skip_end_token=True):
        """
        Convert token indices back to text.
        
        Args:
            ids: List of integer indices
            skip_end_token: Whether to skip <END> token in output
        
        Returns:
            Decoded string
        """
        chars = []
        for idx in ids:
            if skip_end_token and idx == 0:
                continue  # Skip <END> token
            chars.append(self.itos.get(idx, '<UNK>'))
        
        return ''.join(chars)
    
    def save(self, path):
        """
        Save tokenizer vocabulary to file.
        
        Args:
            path: File path (will save as .json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'chars': self.chars,
            'vocab_size': self.vocab_size,
            'stoi': self.stoi,
            'itos': self.itos
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load tokenizer from file.
        
        Args:
            path: File path to .json file
        
        Returns:
            Tokenizer instance with loaded vocabulary
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            chars=data['chars'],
            stoi=data['stoi'],
            itos=data['itos']
        )
    
    def __repr__(self):
        return (f"Tokenizer(vocab_size={self.vocab_size}, "
                f"chars={len(self.chars)}, "
                f"special_tokens={self.special_tokens})")
    
    def __len__(self):
        return self.vocab_size


def get_text_data(path):
    """
    Load and preprocess text data from file.
    
    Handles various text formats and encodings.
    
    Args:
        path: Path to text file
    
    Returns:
        Preprocessed text string
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        # Last resort: read as binary and decode with error handling
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8', errors='replace')
    
    # Basic preprocessing
    # Remove excessive whitespace while preserving structure
    lines = text.split('\n')
    
    # Remove empty lines and very short lines
    lines = [line.strip() for line in lines if len(line.strip()) > 1]
    
    # Join with newlines
    text = '\n'.join(lines)
    
    return text


def create_minimal_vocabulary():
    """
    Create a minimal vocabulary for testing.
    
    Returns:
        Tokenizer with basic ASCII characters
    """
    # Minimal character set for testing
    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?-:;\'"()[] ')
    
    return Tokenizer(chars=chars)


if __name__ == "__main__":
    print("=" * 60)
    print("🔤 TinyLLM Tokenizer Test")
    print("=" * 60)
    print()
    
    # Create tokenizer
    tokenizer = Tokenizer()
    print(f"Empty tokenizer: {tokenizer}")
    
    # Learn from sample text
    sample_text = """
    Once upon a time, there was a small language model.
    It wanted to learn how to speak and write.
    So it studied many books and learned new words.
    Now it can generate text all by itself!
    """
    
    tokenizer.fit(sample_text)
    print(f"\nFitted tokenizer: {tokenizer}")
    print(f"Vocabulary: {tokenizer.chars}")
    
    # Test encoding
    test_text = "Hello, World!"
    encoded = tokenizer.encode(test_text)
    print(f"\nEncoding '{test_text}':")
    print(f"  Tokens: {encoded}")
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoding back: '{decoded}'")
    
    # Test with special tokens
    encoded_with_end = tokenizer.encode("Test", add_end_token=True)
    print(f"\nWith <END> token: {encoded_with_end}")
    print(f"Decoded (skip <END>): '{tokenizer.decode(encoded_with_end)}'")
    print(f"Decoded (include <END>): '{tokenizer.decode(encoded_with_end, skip_end_token=False)}'")
    
    print("\n✅ Tokenizer test complete!")
    print("\nNext steps:")
    print("  1. Create data/input.txt with your training text")
    print("  2. Run train.py to train the model")
