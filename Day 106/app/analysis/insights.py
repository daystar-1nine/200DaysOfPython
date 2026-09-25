"""
Theoretical synthesis and model comparison insights for Day 106: RNNs & Sequential Text Learning.
"""

from typing import Dict, Any


class ModelInsights:
    """Generates comparative analytical summaries of sequential vs non-sequential NLP models."""

    @staticmethod
    def get_architectural_insights() -> Dict[str, str]:
        return {
            "order_sensitivity": (
                "SimpleRNN explicitly processes tokens in time order (h_t = tanh(Wx*x_t + Wh*h_{t-1} + b)), "
                "allowing it to discern syntax, negations ('not bad' vs 'bad not'), and word positions. "
                "In contrast, global average pooling (Day 105) collapses sequence order completely."
            ),
            "gradient_dynamics": (
                "Vanilla RNNs suffer from vanishing and exploding gradients over long sequence steps (BPTT). "
                "Continuous matrix multiplication of Wh over T steps leads to exponential decay or explosion "
                "proportional to the largest eigenvalue of Wh. Gradient clipping was integrated to stabilize training."
            ),
            "computational_tradeoff": (
                "Classical TF-IDF linear baselines train in milliseconds and perform exceptionally well when "
                "classification is driven primarily by keyword indicators. RNNs require step-by-step unrolling "
                "across time, which increases training and inference latency."
            ),
            "lstm_preview": (
                "Day 107 will introduce LSTMs (Long Short-Term Memory), which incorporate additive cell state "
                "highways (C_t) and gating mechanisms (forget, input, output gates) to eliminate vanishing gradients."
            ),
        }
