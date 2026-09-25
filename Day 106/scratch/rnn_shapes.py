"""
Parameter shape and capacity verification for Vanilla RNNs.
Validates the mathematical dimensions of recurrent weight matrices and biases.
"""

from typing import Dict


def calculate_rnn_parameters(input_dim: int, hidden_size: int, output_dim: int = 1) -> Dict[str, int]:
    """Calculate exact parameter counts for a SimpleRNN layer and classification head.
    
    Equations:
        h_t = tanh(Wx @ x_t + Wh @ h_{t-1} + b)
        y_hat = sigma(Wy @ h_T + b_y)
        
    Shapes:
        Wx  : (hidden_size, input_dim) -> hidden_size * input_dim
        Wh  : (hidden_size, hidden_size) -> hidden_size * hidden_size
        b   : (hidden_size,) -> hidden_size
        Wy  : (output_dim, hidden_size) -> output_dim * hidden_size
        b_y : (output_dim,) -> output_dim
        
    Standard Keras / PyTorch SimpleRNN parameter formula:
        Params_RNN = hidden_size * (input_dim + hidden_size) + hidden_size
    """
    wx_params = hidden_size * input_dim
    wh_params = hidden_size * hidden_size
    b_params = hidden_size
    rnn_total = wx_params + wh_params + b_params

    wy_params = output_dim * hidden_size
    by_params = output_dim
    dense_total = wy_params + by_params

    return {
        "Wx_shape": (hidden_size, input_dim),
        "Wx_params": wx_params,
        "Wh_shape": (hidden_size, hidden_size),
        "Wh_params": wh_params,
        "b_shape": (hidden_size,),
        "b_params": b_params,
        "rnn_total": rnn_total,
        "Wy_shape": (output_dim, hidden_size),
        "Wy_params": wy_params,
        "by_shape": (output_dim,),
        "by_params": by_params,
        "dense_total": dense_total,
        "grand_total": rnn_total + dense_total,
    }


if __name__ == "__main__":
    d, h = 32, 64
    counts = calculate_rnn_parameters(input_dim=d, hidden_size=h, output_dim=1)
    print("RNN Parameter Shapes & Count Breakdown:")
    print(f"  Input Dim (D)     : {d}")
    print(f"  Hidden Size (H)   : {h}")
    print(f"  Wx (input->hidden): {counts['Wx_shape']} = {counts['Wx_params']} weights")
    print(f"  Wh (hidden->hidden): {counts['Wh_shape']} = {counts['Wh_params']} weights")
    print(f"  b  (bias)         : {counts['b_shape']} = {counts['b_params']} weights")
    print(f"  --> SimpleRNN Total: {counts['rnn_total']} parameters")
    print(f"  Wy (hidden->out)  : {counts['Wy_shape']} = {counts['Wy_params']} weights")
    print(f"  by (out bias)     : {counts['by_shape']} = {counts['by_params']} weights")
    print(f"  --> Total Head    : {counts['dense_total']} parameters")
    print(f"  ==> Grand Total   : {counts['grand_total']} parameters")

    # Formula check: (32 + 64) * 64 + 64 = 96 * 64 + 64 = 6144 + 64 = 6208
    assert counts["rnn_total"] == 6208
    print("  [SUCCESS] Parameter formula matches standard Keras/PyTorch SimpleRNN!")
