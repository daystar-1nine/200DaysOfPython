"""
Recurrent Neural Network model implementations for Day 106: RNNs & Sequential Text Learning.
Provides:
  1. PyTorch SimpleRNNClassifier (production-ready, verified on Python 3.14).
  2. TensorFlow Keras RNN builder (for environments with TensorFlow support).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class SimpleRNNClassifier(nn.Module):
    """Many-to-One Recurrent Neural Network Text Classifier in PyTorch.
    
    Architecture:
        Input IDs: (Batch, Seq_Len)
            ↓
        Embedding Layer: (Batch, Seq_Len, Embedding_Dim)
            ↓
        Simple RNN (tanh): (Batch, Seq_Len, Hidden_Units)
            ↓
        Final Hidden State Selection (masked to last valid token)
            ↓
        Dropout(p=dropout_1)
            ↓
        Dense Layer (Hidden_Units -> Dense_Units) + ReLU
            ↓
        Dropout(p=dropout_2)
            ↓
        Dense Output (Dense_Units -> 1) -> Logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_units: int = 64,
        dense_units: int = 32,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
        padding_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dense_units = dense_units
        self.padding_idx = padding_idx

        # 1. Trainable Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # 2. Recurrent Layer (tanh activation, batch_first)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_units,
            batch_first=True,
            nonlinearity="tanh"
        )

        # 3. Regularization & Dense Classification Head
        self.dropout_1 = nn.Dropout(p=dropout_1)
        self.fc1 = nn.Linear(hidden_units, dense_units)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=dropout_2)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequences: bool = False,
        return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through Embedding, RNN, and Classification Head.
        
        Args:
            input_ids: Tensor of shape (Batch, Seq_Len) of integer token IDs.
            mask: Optional Tensor of shape (Batch, Seq_Len), 1.0 for valid, 0.0 for pad.
            return_sequences: If True, returns full sequence of hidden states (Batch, Seq_Len, Hidden).
            return_state: If True, returns tuple of (logits, final_hidden_state).
            
        Returns:
            Logits tensor of shape (Batch,) or tuple if return_state is True.
        """
        # (Batch, Seq_Len, Embedding_Dim)
        x = self.embedding(input_ids)

        # RNN Forward: rnn_out is (Batch, Seq_Len, Hidden), h_n is (1, Batch, Hidden)
        rnn_out, h_n = self.rnn(x)

        if return_sequences:
            return rnn_out

        # Select last valid hidden state using mask if provided
        if mask is not None:
            # Length of valid tokens per sequence: (Batch,)
            lengths = mask.sum(dim=1).long().clamp(min=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            # Gather state at index lengths: shape (Batch, Hidden)
            h_final = rnn_out[batch_indices, lengths, :]
        else:
            h_final = h_n.squeeze(0)  # (Batch, Hidden)

        # Classification Head
        drop1 = self.dropout_1(h_final)
        dense1 = self.relu(self.fc1(drop1))
        drop2 = self.dropout_2(dense1)
        logits = self.fc2(drop2).squeeze(-1)  # (Batch,)

        if return_state:
            return logits, h_final

        return logits

    def predict_proba(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Compute class 1 probabilities using Sigmoid."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, mask)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def count_parameters(self) -> Dict[str, int]:
        """Compute parameter counts across model layers."""
        emb_params = sum(p.numel() for p in self.embedding.parameters())
        rnn_params = sum(p.numel() for p in self.rnn.parameters())
        fc1_params = sum(p.numel() for p in self.fc1.parameters())
        fc2_params = sum(p.numel() for p in self.fc2.parameters())
        total = emb_params + rnn_params + fc1_params + fc2_params
        return {
            "embedding": emb_params,
            "rnn": rnn_params,
            "head": fc1_params + fc2_params,
            "total": total
        }


def build_keras_rnn(
    vocab_size: int,
    embedding_dim: int = 64,
    hidden_units: int = 64,
    dense_units: int = 32,
    dropout_1: float = 0.3,
    dropout_2: float = 0.2
):
    """Build the official Keras SimpleRNN Sequential model specified in Section 17.
    
    Raises:
        ImportError: If tensorflow is not installed in the environment.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not available in the current environment. "
            "STATUS: UNVERIFIED — ENVIRONMENT LIMITATION (Python 3.14 lacks TF wheels)."
        ) from e

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        ),
        tf.keras.layers.SimpleRNN(hidden_units),
        tf.keras.layers.Dropout(dropout_1),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model
