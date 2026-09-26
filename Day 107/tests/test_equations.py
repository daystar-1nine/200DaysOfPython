import pytest
import numpy as np
from scratch.lstm_equations import ScratchLSTMCell, sigmoid, tanh

def test_sigmoid():
    assert np.isclose(sigmoid(0), 0.5)
    assert np.isclose(sigmoid(100), 1.0)
    assert np.isclose(sigmoid(-100), 0.0)

def test_tanh():
    assert np.isclose(tanh(0), 0.0)
    assert np.isclose(tanh(100), 1.0)
    assert np.isclose(tanh(-100), -1.0)

def test_scratch_lstm_init():
    cell = ScratchLSTMCell(input_size=10, hidden_size=20)
    assert cell.W_f.shape == (30, 20)
    assert cell.b_f.shape == (1, 20)
    # Check forget bias init
    assert np.all(cell.b_f == 1.0)

@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_scratch_lstm_forward(batch_size):
    input_size = 8
    hidden_size = 12
    cell = ScratchLSTMCell(input_size, hidden_size)
    
    x_t = np.random.randn(batch_size, input_size)
    h_prev = np.zeros((batch_size, hidden_size))
    c_prev = np.zeros((batch_size, hidden_size))
    
    h_t, c_t, cache = cell.forward(x_t, h_prev, c_prev)
    
    assert h_t.shape == (batch_size, hidden_size)
    assert c_t.shape == (batch_size, hidden_size)
    assert len(cache) == 9

@pytest.mark.parametrize("i", range(15))
def test_scratch_lstm_repeat_shapes(i):
    input_size = i + 1
    hidden_size = i + 2
    cell = ScratchLSTMCell(input_size, hidden_size)
    x_t = np.random.randn(2, input_size)
    h_prev = np.zeros((2, hidden_size))
    c_prev = np.zeros((2, hidden_size))
    h_t, _, _ = cell.forward(x_t, h_prev, c_prev)
    assert h_t.shape == (2, hidden_size)
