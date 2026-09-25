"""
Sequence padding, truncation, and masking module for Day 106: RNNs & Sequential Text Learning.
"""

from typing import List, Tuple
import numpy as np


class SequencePadder:
    """Standardizes sequences to fixed length with padding, truncation, and boolean masks."""

    @staticmethod
    def pad_sequence(
        sequence: List[int],
        max_length: int,
        padding: str = "post",
        truncating: str = "post",
        pad_value: int = 0
    ) -> List[int]:
        """Pad or truncate a single integer sequence to max_length.
        
        Args:
            sequence: List of integer token IDs.
            max_length: Target sequence length.
            padding: 'pre' or 'post'.
            truncating: 'pre' or 'post'.
            pad_value: Integer ID for padding token.
            
        Returns:
            List of length max_length.
        """
        # Truncation
        if len(sequence) > max_length:
            if truncating == "post":
                seq = sequence[:max_length]
            elif truncating == "pre":
                seq = sequence[-max_length:]
            else:
                raise ValueError(f"Invalid truncating option: {truncating}. Choose 'pre' or 'post'.")
        else:
            seq = list(sequence)

        # Padding
        pad_len = max_length - len(seq)
        if pad_len > 0:
            pads = [pad_value] * pad_len
            if padding == "post":
                seq = seq + pads
            elif padding == "pre":
                seq = pads + seq
            else:
                raise ValueError(f"Invalid padding option: {padding}. Choose 'pre' or 'post'.")

        return seq

    @classmethod
    def pad_batch(
        cls,
        sequences: List[List[int]],
        max_length: int,
        padding: str = "post",
        truncating: str = "post",
        pad_value: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad a batch of sequences and generate corresponding binary mask.
        
        Args:
            sequences: List of integer ID sequences.
            max_length: Target length.
            padding: 'pre' or 'post'.
            truncating: 'pre' or 'post'.
            pad_value: Padding ID.
            
        Returns:
            Tuple of:
                padded_array: (batch_size, max_length) int64
                mask_array: (batch_size, max_length) float32 (1.0 for valid, 0.0 for pad)
        """
        padded_list = [
            cls.pad_sequence(seq, max_length, padding, truncating, pad_value)
            for seq in sequences
        ]
        padded_arr = np.array(padded_list, dtype=np.int64)
        mask_arr = (padded_arr != pad_value).astype(np.float32)
        return padded_arr, mask_arr
