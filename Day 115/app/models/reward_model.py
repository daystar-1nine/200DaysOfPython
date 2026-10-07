"""
Reward Model architecture and Bradley-Terry preference loss for Day 115.
Transforms contextual representations from MiniGPT backbone into a scalar reward
evaluating human preference ranking between response alternatives.
"""
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import ModelConfig, RewardModelConfig
from app.models.minigpt import MiniGPTBackbone


class RewardModel(nn.Module):
    """
    Reward Model that scores input sequences with a scalar preference rating.
    Uses MiniGPT Transformer backbone + representation pooling + linear reward projection.
    """
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        reward_config: Optional[RewardModelConfig] = None
    ):
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.reward_config = reward_config or RewardModelConfig()
        self.config = self.model_config

        self.backbone = MiniGPTBackbone(self.model_config)

        # Reward prediction head
        if self.reward_config.head_hidden_dim is not None:
            self.reward_head = nn.Sequential(
                nn.Linear(self.model_config.embed_dim, self.reward_config.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.reward_config.dropout),
                nn.Linear(self.reward_config.head_hidden_dim, 1, bias=False)
            )
        else:
            self.reward_head = nn.Linear(self.model_config.embed_dim, 1, bias=False)

    def pool_representation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pools sequence hidden states [B, T, D] into a single representation [B, D].
        Supports:
          - 'last': representation at the last valid (non-padding) token
          - 'mean': mean-pooled representation across non-padding tokens
        """
        B, T, D = hidden_states.size()

        if attention_mask is None:
            # Fallback to last token
            return hidden_states[:, -1, :]

        if self.reward_config.pooling_method == "mean":
            # Masked average
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return sum_hidden / lengths

        # Default 'last' token pooling
        lengths = attention_mask.sum(dim=-1).clamp(min=1)  # [B]
        last_indices = (lengths - 1).long()  # 0-indexed position of last valid token

        batch_indices = torch.arange(B, device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_indices, :]  # [B, D]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes scalar reward scores for given sequences.
        Returns:
          rewards: [B] scalar float tensor
        """
        hidden_states = self.backbone(input_ids, attention_mask=attention_mask)
        pooled = self.pool_representation(hidden_states, attention_mask=attention_mask)
        rewards = self.reward_head(pooled).squeeze(-1)  # [B]
        return rewards

    def forward_pairwise(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes rewards for chosen and rejected responses in parallel.
        Returns:
          (r_chosen, r_rejected) each of shape [B]
        """
        r_chosen = self.forward(chosen_ids, chosen_mask)
        r_rejected = self.forward(rejected_ids, rejected_mask)
        return r_chosen, r_rejected


def compute_reward_loss(
    r_chosen: torch.Tensor,
    r_rejected: torch.Tensor,
    margin: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Bradley-Terry pairwise preference ranking loss:
      L = -log sigmoid(r_chosen - r_rejected - margin).mean()

    Returns:
      (loss, telemetry_metrics)
    """
    diff = r_chosen - r_rejected - margin
    loss = -F.logsigmoid(diff).mean()

    with torch.no_grad():
        accuracy = (r_chosen > r_rejected).float().mean().item()
        mean_margin = (r_chosen - r_rejected).mean().item()
        r_c_mean = r_chosen.mean().item()
        r_r_mean = r_rejected.mean().item()

    metrics = {
        "reward_loss": float(loss.item()),
        "accuracy": float(accuracy),
        "mean_margin": float(mean_margin),
        "chosen_reward": float(r_c_mean),
        "rejected_reward": float(r_r_mean)
    }
    return loss, metrics
