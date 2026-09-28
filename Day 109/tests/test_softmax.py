import pytest
import numpy as np
from scratch.softmax_numpy import softmax

def test_softmax_output_shape():
    x = np.random.randn(4, 8)
    out = softmax(x, axis=-1)
    assert out.shape == (4, 8)

def test_softmax_values_between_zero_and_one():
    x = np.random.randn(10, 15)
    out = softmax(x, axis=-1)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)

def test_softmax_rows_sum_to_one():
    x = np.random.randn(5, 7)
    out = softmax(x, axis=-1)
    row_sums = np.sum(out, axis=-1)
    assert np.allclose(row_sums, 1.0)

def test_softmax_numerical_stability_large_positive():
    x = np.array([1000.0, 1001.0, 999.0])
    out = softmax(x)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.isclose(np.sum(out), 1.0)
    assert out[1] > out[0] > out[2]

def test_softmax_numerical_stability_large_negative():
    x = np.array([-1000.0, -1005.0, -995.0])
    out = softmax(x)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.isclose(np.sum(out), 1.0)
    assert out[2] > out[0] > out[1]

def test_softmax_single_element():
    x = np.array([5.0])
    out = softmax(x)
    assert np.isclose(out[0], 1.0)

def test_softmax_equal_inputs():
    x = np.array([2.0, 2.0, 2.0, 2.0])
    out = softmax(x)
    assert np.allclose(out, 0.25)

@pytest.mark.parametrize("shape", [(3, 4), (2, 5, 6), (1, 10)])
def test_softmax_multidimensional(shape):
    x = np.random.randn(*shape)
    out = softmax(x, axis=-1)
    assert out.shape == shape
    sums = np.sum(out, axis=-1)
    assert np.allclose(sums, 1.0)

def test_softmax_axis_zero():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = softmax(x, axis=0)
    col_sums = np.sum(out, axis=0)
    assert np.allclose(col_sums, 1.0)
