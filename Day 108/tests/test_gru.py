import pytest
import numpy as np
import torch
from scratch.gru_equations import sigmoid, tanh, gru_step, gru_forward_sequence
from app.models.gru import GRUClassifier
from app.models.bigru import BiGRUClassifier
from app.models.stacked_gru import StackedGRUClassifier

# --- GRU Mathematics Tests ---

def test_sigmoid_numerical_behavior():
    # Range check [0, 1]
    vals = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    sig = sigmoid(vals)
    assert np.all(sig >= 0.0)
    assert np.all(sig <= 1.0)
    assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)
    assert np.isclose(sigmoid(np.array([500.0]))[0], 1.0)
    assert np.isclose(sigmoid(np.array([-500.0]))[0], 0.0)

def test_tanh_output_range():
    vals = np.array([-10.0, 0.0, 10.0])
    res = tanh(vals)
    assert np.all(res >= -1.0) and np.all(res <= 1.0)
    assert np.isclose(res[1], 0.0)

def test_gru_step_gate_shapes_and_states():
    input_dim = 16
    hidden_dim = 32
    
    Wz = np.zeros((hidden_dim, input_dim))
    Uz = np.zeros((hidden_dim, hidden_dim))
    bz = np.zeros(hidden_dim)
    
    Wr = np.zeros((hidden_dim, input_dim))
    Ur = np.zeros((hidden_dim, hidden_dim))
    br = np.zeros(hidden_dim)
    
    Wh = np.zeros((hidden_dim, input_dim))
    Uh = np.zeros((hidden_dim, hidden_dim))
    bh = np.zeros(hidden_dim)
    
    x_t = np.random.randn(input_dim)
    h_prev = np.random.randn(hidden_dim)
    
    h_t, gates = gru_step(x_t, h_prev, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh)
    
    # Assert shapes
    assert h_t.shape == (hidden_dim,)
    assert gates["z_t"].shape == (hidden_dim,)
    assert gates["r_t"].shape == (hidden_dim,)
    assert gates["h_tilde"].shape == (hidden_dim,)
    
    # Assert values for all-zero weights:
    # z_t = sig(0) = 0.5, r_t = sig(0) = 0.5, h_tilde = tanh(0) = 0.0
    # h_t = 0.5 * h_prev + 0.5 * 0 = 0.5 * h_prev
    assert np.allclose(gates["z_t"], 0.5)
    assert np.allclose(gates["r_t"], 0.5)
    assert np.allclose(gates["h_tilde"], 0.0)
    assert np.allclose(h_t, 0.5 * h_prev)

def test_gru_deterministic_calculation():
    # Verify deterministic calculation across repeated calls
    np.random.seed(123)
    input_dim = 8
    hidden_dim = 16
    Wz = np.random.randn(hidden_dim, input_dim)
    Uz = np.random.randn(hidden_dim, hidden_dim)
    bz = np.random.randn(hidden_dim)
    Wr, Ur, br = Wz.copy(), Uz.copy(), bz.copy()
    Wh, Uh, bh = Wz.copy(), Uz.copy(), bz.copy()
    
    x = np.random.randn(input_dim)
    h0 = np.random.randn(hidden_dim)
    
    h_a, _ = gru_step(x, h0, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh)
    h_b, _ = gru_step(x, h0, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh)
    assert np.array_equal(h_a, h_b)

def test_gru_forward_sequence_shape():
    seq_len = 15
    input_dim = 10
    hidden_dim = 20
    X = np.random.randn(seq_len, input_dim)
    h_0 = np.zeros(hidden_dim)
    
    weights = {
        "Wz": np.random.randn(hidden_dim, input_dim) * 0.1,
        "Uz": np.random.randn(hidden_dim, hidden_dim) * 0.1,
        "bz": np.zeros(hidden_dim),
        "Wr": np.random.randn(hidden_dim, input_dim) * 0.1,
        "Ur": np.random.randn(hidden_dim, hidden_dim) * 0.1,
        "br": np.zeros(hidden_dim),
        "Wh": np.random.randn(hidden_dim, input_dim) * 0.1,
        "Uh": np.random.randn(hidden_dim, hidden_dim) * 0.1,
        "bh": np.zeros(hidden_dim),
    }
    
    all_h, h_final = gru_forward_sequence(X, h_0, weights)
    assert all_h.shape == (seq_len, hidden_dim)
    assert h_final.shape == (hidden_dim,)
    assert np.allclose(all_h[-1], h_final)

# --- PyTorch GRU Models Tests ---

def test_gru_classifier_forward_and_params():
    model = GRUClassifier(vocab_size=100, embedding_dim=16, hidden_dim=32, dense_dim=16)
    x = torch.randint(0, 100, (4, 10))
    probs = model(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert model.count_parameters() > 0

def test_bigru_classifier_forward_and_params():
    model = BiGRUClassifier(vocab_size=100, embedding_dim=16, hidden_dim=32, dense_dim=16)
    x = torch.randint(0, 100, (4, 10))
    probs = model(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert model.count_parameters() > 0

def test_stacked_gru_classifier_forward_and_params():
    model = StackedGRUClassifier(vocab_size=100, embedding_dim=16, hidden_dim1=32, hidden_dim2=16)
    x = torch.randint(0, 100, (4, 10))
    probs = model(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert model.count_parameters() > 0

@pytest.mark.parametrize("batch_size, seq_len", [
    (1, 5),
    (2, 20),
    (8, 30)
])
def test_gru_classifier_various_batch_sizes(batch_size, seq_len):
    model = GRUClassifier(vocab_size=50, embedding_dim=8, hidden_dim=16)
    x = torch.randint(0, 50, (batch_size, seq_len))
    probs = model(x)
    assert probs.shape == (batch_size,)
