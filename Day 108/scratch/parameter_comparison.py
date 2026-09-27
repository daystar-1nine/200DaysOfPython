"""
Theoretical and Programmatic Parameter Count Comparison between RNN, GRU, and LSTM.
Verifies Practical Task 2 and Section 16 from Day 108.
"""
import torch
import torch.nn as nn

def manual_param_counts(input_size: int = 32, hidden_size: int = 64):
    """
    Computes theoretical parameter counts using canonical gating formulations.
    """
    base_block = (input_size * hidden_size) + (hidden_size * hidden_size) + hidden_size
    
    rnn_canonical = 1 * base_block
    gru_canonical = 3 * base_block
    lstm_canonical = 4 * base_block
    
    return {
        "SimpleRNN": rnn_canonical,
        "GRU": gru_canonical,
        "LSTM": lstm_canonical
    }

def pytorch_param_counts(input_size: int = 32, hidden_size: int = 64):
    """
    Verifies parameter counts using PyTorch recurrent layers (PyTorch stores 2 separate bias vectors per gate).
    """
    rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    count_params = lambda mod: sum(p.numel() for p in mod.parameters())
    
    return {
        "SimpleRNN": count_params(rnn),
        "GRU": count_params(gru),
        "LSTM": count_params(lstm)
    }

def print_parameter_comparison():
    input_size = 32
    hidden_size = 64
    
    manual = manual_param_counts(input_size, hidden_size)
    pt = pytorch_param_counts(input_size, hidden_size)
    
    print(f"=== Parameter Count Analysis (Input={input_size}, Hidden={hidden_size}) ===")
    print(f"{'Architecture':<12} | {'Canonical Form':<16} | {'PyTorch Layer':<16} | {'Gates'}")
    print("-" * 58)
    print(f"{'SimpleRNN':<12} | {manual['SimpleRNN']:<16} | {pt['SimpleRNN']:<16} | 1 (None, linear tanh)")
    print(f"{'GRU':<12} | {manual['GRU']:<16} | {pt['GRU']:<16} | 3 (Update, Reset, Candidate)")
    print(f"{'LSTM':<12} | {manual['LSTM']:<16} | {pt['LSTM']:<16} | 4 (Forget, Input, Candidate, Output)")
    print("-" * 58)
    print("\nArchitectural Insights:")
    print("1. GRU eliminates the separate cell state C_t and merges Forget/Input decisions into a single Update gate z_t.")
    print("2. GRU has exactly 3 sets of transformations vs LSTM's 4 sets -> ~25% parameter reduction.")
    print("3. Vanilla RNN has only 1 set of transformations, but suffers severely from vanishing gradients.")

if __name__ == "__main__":
    print_parameter_comparison()
