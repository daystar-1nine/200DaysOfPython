import pytest
import numpy as np
from pathlib import Path
from scratch.self_attention_numpy import plot_attention_matrix
from app.visualization.attention_heatmap import plot_single_message_attention
from app.analysis.entropy import compute_attention_entropy

def test_plot_attention_matrix_runs_and_saves(tmp_path):
    weights = np.array([
        [0.7, 0.3],
        [0.2, 0.8]
    ])
    tokens = ["tokenA", "tokenB"]
    out_file = tmp_path / "matrix.png"
    plot_attention_matrix(weights, tokens, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

def test_plot_single_message_attention_runs_and_saves(tmp_path):
    tokens = ["free", "entry", "win"]
    weights = np.array([0.5, 0.3, 0.2])
    out_file = tmp_path / "single_heatmap.png"
    plot_single_message_attention(tokens, weights, title="Test Heatmap", save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

def test_single_message_attention_mismatched_lengths(tmp_path):
    tokens = ["a", "b", "c", "d"]
    weights = np.array([0.5, 0.5])
    out_file = tmp_path / "mismatch.png"
    # Should safely truncate to min(len(tokens), len(weights))
    plot_single_message_attention(tokens, weights, title="Mismatch Test", save_path=out_file)
    assert out_file.exists()

def test_compute_attention_entropy_single_token():
    weights = np.array([1.0])
    ent = compute_attention_entropy(weights)
    # log(1.0) = 0.0 -> entropy should be 0.0
    assert np.isclose(ent, 0.0, atol=1e-4)

@pytest.mark.parametrize("n_tokens", [2, 4, 8, 16])
def test_attention_entropy_scales_with_uniform_distribution(n_tokens):
    weights = np.ones(n_tokens) / n_tokens
    ent = compute_attention_entropy(weights)
    expected_ent = np.log(n_tokens)
    assert np.isclose(ent, expected_ent, atol=1e-3)
