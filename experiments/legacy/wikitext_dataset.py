#!/usr/bin/env python3
"""
WikiText-103 dataset loader for MoE training with synthetic gradients.
Uses HuggingFace datasets library for reliable data loading.
Aligned with OLMoE tokenizer for proper vocabulary matching.
Memory-efficient streaming implementation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional, Iterator
from datasets import load_dataset
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingWikiText103Dataset(Dataset):
    """Memory-efficient WikiText-103 dataset with streaming."""

    def __init__(self, split: str = "train", seq_len: int = 128,
                 tokenizer_name: str = "./moe_app/models/olmo_sharded/tokenizer",
                 max_samples: Optional[int] = None,
                 buffer_size: int = 1000):
        """
        Initialize streaming WikiText-103 dataset.
        
        Args:
            split: "train", "validation", or "test"
            seq_len: Sequence length for training
            tokenizer_name: HuggingFace tokenizer to use (OLMoE tokenizer)
            max_samples: Maximum number of samples (for debugging)
            buffer_size: Number of sequences to keep in memory buffer
        """
        self.split = split
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.buffer_size = buffer_size

        # Load tokenizer (prefer local path/cache, then online)
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        try:
            if Path(tokenizer_name).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            else:
                try:
                    # Prefer local HF cache if present
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name, trust_remote_code=True, local_files_only=True
                    )
                except Exception as local_err:
                    logger.warning(f"Tokenizer cache miss for {tokenizer_name}: {local_err}. Trying online.")
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Tokenizer load failed for {tokenizer_name}: {e}. Falling back to GPT-2")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize streaming dataset
        self.dataset = self._load_streaming_dataset()

        # Initialize sequence buffer
        self.sequence_buffer = []
        self.buffer_index = 0
        self.total_sequences = 0

        # Pre-fill buffer
        self._fill_buffer()

        logger.info(f"Streaming WikiText-103 {split}: buffer size {buffer_size}, seq_len {seq_len}")

    def _load_streaming_dataset(self):
        """Load WikiText-103 as streaming dataset."""
        logger.info("Loading WikiText-103 as streaming dataset...")

        try:
            # Load with streaming=True for memory efficiency
            dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", streaming=True)

            # Map split names
            split_map = {
                "train": "train",
                "validation": "validation",
                "test": "test"
            }

            if self.split not in split_map:
                raise ValueError(f"Invalid split: {self.split}")

            return dataset[split_map[self.split]]

        except Exception as e:
            logger.warning(f"Failed to load WikiText-103 from HuggingFace: {e}")
            logger.info("Falling back to synthetic dataset...")
            return self._create_synthetic_streaming_dataset()

    def _create_synthetic_streaming_dataset(self):
        """Create synthetic streaming dataset for testing."""
        synthetic_texts = [
            "Robert Edward Lee was an American Confederate general who served in the Confederate States Army during the American Civil War.",
            "He was the son of Henry Lee III and Anne Hill Carter Lee. His father had been a major general in the Continental Army.",
            "Lee was born at Stratford Hall Plantation in Westmoreland County, Virginia, and was a member of the state's aristocracy.",
            "He was related to the state's governor, Thomas Lee, and was a descendant of King Robert II of Scotland through the Earls of Crawford.",
            "His father, Henry Lee III, was a Revolutionary War officer, governor of Virginia, and member of the Continental Congress.",
            "Lee graduated from the United States Military Academy at West Point in 1829. He served in the United States Army for 32 years.",
            "During which time he fought in the Mexican–American War and served as superintendent of the United States Military Academy.",
            "He was also involved in suppressing John Brown's raid on Harpers Ferry and capturing Brown and his co-conspirators.",
            "When Virginia declared its secession from the Union in April 1861, Lee chose to follow his home state.",
            "Despite his personal desire for the country to remain intact and an offer of a senior Union command.",
            "During the first year of the Civil War, he served as a senior military adviser to Confederate President Jefferson Davis.",
            "Once he took command of the main field army in 1862 he soon emerged as a shrewd tactician and battlefield commander.",
            "Winning most of his battles, all against far superior Union armies.",
            "Lee's two major strategic offensives into Union territory both ended in defeat.",
            "His aggressive tactics, especially at the Battle of Gettysburg, which resulted in heavy casualties.",
            "At a time when the Confederacy had a shortage of manpower, have come under criticism in recent years.",
            "Lee surrendered the remnant of his army to Ulysses S. Grant at Appomattox Court House on April 9, 1865.",
            "Since Confederate President Jefferson Davis had fled the capital of Richmond, Lee was the highest-ranking Confederate general remaining.",
            "And Grant accepted his surrender. By this time, Lee had assumed supreme command of the remaining Southern forces.",
            "Other Confederate forces swiftly capitulated after his surrender.",
            "Lee rejected the proposal of a sustained insurgency against the Union and called for reconciliation between the two sides.",
            "After the war, as President of what is now Washington and Lee University, he supported President Andrew Johnson's program.",
            "Of Reconstruction and intersectional friendship, while opposing the Radical Republican proposals.",
            "To give freed slaves the vote and take the vote away from ex-Confederates.",
            "He urged them to rethink their position between the North and South, and the reintegration of former Confederates.",
            "Into the nation's political life. Lee became the great Southern hero of the War, a postwar icon of the Lost Cause.",
            "Of the Confederacy to some. But his popularity grew even in the North, especially after his death in 1870.",
            "He remains an iconic figure of American military leadership."
        ]

        # Repeat texts for more data
        if self.split == "train":
            synthetic_texts = synthetic_texts * 100
        elif self.split == "validation":
            synthetic_texts = synthetic_texts * 20
        else:  # test
            synthetic_texts = synthetic_texts * 10

        # Create a simple iterator
        class SyntheticIterator:
            def __iter__(self):
                for text in synthetic_texts:
                    yield {"text": text}

        return SyntheticIterator()

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize a single text piece."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _create_sequences_from_tokens(self, tokens: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create sequences from a list of tokens."""
        sequences = []

        for i in range(0, len(tokens) - self.seq_len, self.seq_len // 2):  # 50% overlap
            if i + self.seq_len + 1 >= len(tokens):
                break

            # Input sequence
            input_ids = tokens[i:i + self.seq_len]
            # Target sequence (shifted by 1)
            target_ids = tokens[i + 1:i + self.seq_len + 1]

            # Convert to tensors
            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            target_tensor = torch.tensor(target_ids, dtype=torch.long)

            sequences.append((input_tensor, target_tensor))

        return sequences

    def _fill_buffer(self):
        """Fill the sequence buffer with new sequences."""
        if len(self.sequence_buffer) >= self.buffer_size:
            return  # Buffer is full

        logger.info(f"Filling sequence buffer... (current: {len(self.sequence_buffer)})")

        # Collect tokens from dataset
        all_tokens = []
        token_count = 0
        target_tokens = self.buffer_size * self.seq_len * 2  # Estimate needed tokens

        try:
            for example in self.dataset:
                text = example["text"]
                if text.strip():  # Skip empty text
                    tokens = self._tokenize_text(text)
                    all_tokens.extend(tokens)
                    token_count += len(tokens)

                    if token_count >= target_tokens:
                        break
        except Exception as e:
            logger.warning(f"Error reading from dataset: {e}")
            # Use synthetic data as fallback
            all_tokens = self._tokenize_text("Robert Edward Lee was an American Confederate general. " * 1000)

        # Create sequences from collected tokens
        new_sequences = self._create_sequences_from_tokens(all_tokens)

        # Add to buffer
        self.sequence_buffer.extend(new_sequences)
        self.total_sequences += len(new_sequences)

        # Limit buffer size
        if len(self.sequence_buffer) > self.buffer_size:
            self.sequence_buffer = self.sequence_buffer[:self.buffer_size]

        logger.info(f"Buffer filled: {len(self.sequence_buffer)} sequences, total: {self.total_sequences}")

        # Force garbage collection
        gc.collect()

    def __len__(self) -> int:
        if self.max_samples:
            return min(self.max_samples, self.total_sequences)
        return self.total_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check if we need to refill buffer
        if self.buffer_index >= len(self.sequence_buffer):
            self.buffer_index = 0
            self._fill_buffer()

        # Get sequence from buffer
        sequence = self.sequence_buffer[self.buffer_index]
        self.buffer_index += 1

        return sequence

class WikiText103DataLoader:
    """Memory-efficient data loader for WikiText-103."""

    def __init__(self, split: str = "train", batch_size: int = 32, seq_len: int = 128,
                 tokenizer_name: str = "./moe_app/models/olmo_sharded/tokenizer",
                 max_samples: Optional[int] = None,
                 buffer_size: int = 1000,
                 num_workers: int = 0):  # Reduced workers for memory efficiency
        """
        Initialize memory-efficient WikiText-103 data loader.
        
        Args:
            split: Dataset split
            batch_size: Batch size for training
            seq_len: Sequence length
            tokenizer_name: Tokenizer to use (OLMoE tokenizer)
            max_samples: Maximum samples (for debugging)
            buffer_size: Number of sequences to keep in memory
            num_workers: Number of worker processes (0 for memory efficiency)
        """
        self.split = split
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer_name = tokenizer_name

        # Create dataset
        self.dataset = StreamingWikiText103Dataset(
            split=split,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            max_samples=max_samples,
            buffer_size=buffer_size
        )

        # Create data loader with minimal workers for memory efficiency
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,  # Disable for memory efficiency
            drop_last=True
        )

        # Get vocabulary size
        self.vocab_size = self.dataset.tokenizer.vocab_size
        logger.info(f"Vocabulary size: {self.vocab_size}")

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_tokenizer(self):
        return self.dataset.tokenizer

def create_wikitext_dataloaders(batch_size: int = 32, seq_len: int = 128,
                               tokenizer_name: str = "./moe_app/models/olmo_sharded/tokenizer",
                               max_train_samples: Optional[int] = None,
                               max_val_samples: Optional[int] = None,
                               buffer_size: int = 1000) -> Tuple[WikiText103DataLoader, WikiText103DataLoader]:
    """Create memory-efficient training and validation dataloaders for WikiText-103."""

    train_loader = WikiText103DataLoader(
        split="train",
        batch_size=batch_size,
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        max_samples=max_train_samples,
        buffer_size=buffer_size
    )

    val_loader = WikiText103DataLoader(
        split="validation",
        batch_size=batch_size,
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        max_samples=max_val_samples,
        buffer_size=buffer_size // 4  # Smaller buffer for validation
    )

    return train_loader, val_loader

def test_wikitext_loader():
    """Test the memory-efficient WikiText-103 data loader."""
    logger.info("Testing memory-efficient WikiText-103 data loader with OLMoE tokenizer...")

    # Create small test loaders with minimal buffer
    train_loader, val_loader = create_wikitext_dataloaders(
        batch_size=4,
        seq_len=64,
        max_train_samples=50,
        max_val_samples=20,
        buffer_size=100  # Small buffer for testing
    )

    # Test training data
    logger.info(f"Training batches: {len(train_loader)}")
    for i, (tokens, targets) in enumerate(train_loader):
        logger.info(f"Batch {i}: tokens shape {tokens.shape}, targets shape {targets.shape}")
        logger.info(f"Sample tokens: {tokens[0][:10]}")
        logger.info(f"Sample targets: {targets[0][:10]}")

        # Decode sample
        sample_text = train_loader.get_tokenizer().decode(tokens[0].tolist())
        logger.info(f"Sample text: {sample_text[:100]}...")
        break

    # Test validation data
    logger.info(f"Validation batches: {len(val_loader)}")
    for i, (tokens, targets) in enumerate(val_loader):
        logger.info(f"Val batch {i}: tokens shape {tokens.shape}, targets shape {targets.shape}")
        break

    logger.info("Memory-efficient WikiText-103 data loader test completed!")

if __name__ == "__main__":
    test_wikitext_loader()
