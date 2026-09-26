import torch
import torch.nn as nn
from typing import Optional

class PyTorchLSTMClassifier(nn.Module):
    """
    Standard, Bidirectional, and Stacked LSTM Implementation in PyTorch.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout_rate: float = 0.5,
        pad_idx: int = 0
    ):
        super().__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0 # Dropout in PyTorch LSTM only applies between layers
        )
        
        # 3. Dropout and Fully Connected Layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # If bidirectional, the hidden state size is doubled
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Linear(lstm_out_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        text shape: (batch_size, sequence_length)
        """
        # (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(text)
        
        # (batch_size, sequence_length, hidden_dim * num_directions)
        # hidden, cell shapes: (num_layers * num_directions, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Extract the final hidden state
        # If bidirectional, we concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            # hidden[-2] is the final forward hidden state
            # hidden[-1] is the final backward hidden state
            final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # hidden[-1] is the final hidden state of the top layer
            final_hidden = hidden[-1, :, :]
            
        final_hidden = self.dropout(final_hidden)
        
        # (batch_size, output_dim)
        logits = self.fc(final_hidden)
        
        # (batch_size,) - Squeeze if output_dim is 1
        return self.sigmoid(logits).squeeze(1)


# TensorFlow / Keras Builders wrapped in ImportError blocks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    
    def build_tf_lstm(vocab_size: int, embedding_dim: int, hidden_dim: int, max_length: int) -> Sequential:
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            LSTM(hidden_dim),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_tf_bidirectional_lstm(vocab_size: int, embedding_dim: int, hidden_dim: int, max_length: int) -> Sequential:
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            Bidirectional(LSTM(hidden_dim)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_tf_stacked_lstm(vocab_size: int, embedding_dim: int, hidden_dim: int, max_length: int) -> Sequential:
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            LSTM(hidden_dim, return_sequences=True),
            Dropout(0.5),
            LSTM(hidden_dim),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model

except ImportError:
    # TensorFlow is not available, which is expected in this Python 3.14 environment
    def build_tf_lstm(*args, **kwargs):
        raise ImportError("TensorFlow is not installed in this environment.")
    def build_tf_bidirectional_lstm(*args, **kwargs):
        raise ImportError("TensorFlow is not installed in this environment.")
    def build_tf_stacked_lstm(*args, **kwargs):
        raise ImportError("TensorFlow is not installed in this environment.")
