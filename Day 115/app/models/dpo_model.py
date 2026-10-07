"""
Direct Preference Optimization (DPO) model and response log-probability engine for Day 115.
Implements the DPO loss function from Rafailov et al. (2023):
  L_DPO = -E[log sigma(beta * [log(pi(y_c|x) / pi_ref(y_c|x)) - log(pi(y_r|x) / pi_ref(y_r|x))])]
"""
import copy
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import ModelConfig, DPOTrainingConfig, GenerationConfig
from app.models.minigpt import MiniGPTBackbone


class MiniGPTChat(nn.Module):
    """
    Autoregressive MiniGPT conversational language model with Pre-LN Transformer blocks,
    tied embeddings, and response log-probability calculation for DPO alignment.
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.backbone = MiniGPTBackbone(self.config)

        # Language modeling head
        self.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size, bias=False)
        if self.config.tie_weights:
            self.lm_head.weight = self.backbone.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns:
          logits: [B, T, V]
          loss: Optional cross-entropy scalar loss if labels are provided
        """
        hidden = self.backbone(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Shift tokens: logits at pos t predict label at pos t+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def get_response_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes total sequence log-probabilities sum_{t in response} log P(y_t | x, y_{<t})
        strictly over active (non -100) response tokens.

        Returns:
          response_log_probs: [B] float tensor
        """
        if labels is None:
            raise ValueError("Labels are required to compute response-only log probabilities.")

        logits, _ = self.forward(input_ids, attention_mask=attention_mask)

        # Shift tokens
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Token-level cross entropy with ignore_index=-100 (returns 0.0 at -100 positions)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100
        )
        per_token_loss = per_token_loss.view(shift_labels.size(0), shift_labels.size(1))

        # Sum negative loss across sequence: -cross_entropy = log P(y_t)
        response_log_probs = -per_token_loss.sum(dim=-1)  # [B]
        return response_log_probs

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        stop_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, top-p,
        and localized repetition penalty.
        """
        self.eval()
        cfg = generation_config or GenerationConfig()
        generated = prompt_ids.clone()
        prompt_len = prompt_ids.size(1)

        for _ in range(cfg.max_new_tokens):
            cond = generated if generated.size(1) <= self.config.context_length else generated[:, -self.config.context_length:]
            logits, _ = self.forward(cond)
            next_logits = logits[:, -1, :].clone()

            # Apply localized repetition penalty to recently emitted response tokens
            if cfg.repetition_penalty != 1.0 and generated.size(1) > prompt_len:
                for b in range(generated.size(0)):
                    recent = set(generated[b, max(prompt_len, generated.size(1) - 20):].tolist())
                    for tid in recent:
                        if next_logits[b, tid] > 0:
                            next_logits[b, tid] /= cfg.repetition_penalty
                        else:
                            next_logits[b, tid] *= cfg.repetition_penalty

            if not cfg.do_sample or cfg.temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(cfg.temperature, 1e-4)

                # Top-K
                if cfg.top_k > 0:
                    v, _ = torch.topk(scaled, min(cfg.top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = float("-inf")

                # Top-P (Nucleus)
                if cfg.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > cfg.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = torch.zeros_like(scaled, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                    scaled[indices_to_remove] = float("-inf")

                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if stop_token_id is not None and (next_token == stop_token_id).all():
                break

        return generated


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    loss_type: str = "sigmoid"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes DPO loss and implicit reward telemetry.

    Parameters:
      policy_chosen_logps: log pi_theta(y_c | x) [B]
      policy_rejected_logps: log pi_theta(y_r | x) [B]
      reference_chosen_logps: log pi_ref(y_c | x) [B]
      reference_rejected_logps: log pi_ref(y_r | x) [B]
      beta: temperature hyperparameter
      loss_type: 'sigmoid' (standard DPO), 'hinge', or 'ipo'

    Returns:
      (loss, metrics_dict)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios  # [B]

    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = torch.relu(1.0 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"Unsupported DPO loss_type: {loss_type}")

    loss = losses.mean()

    with torch.no_grad():
        chosen_implicit_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_implicit_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_margin = chosen_implicit_rewards - rejected_implicit_rewards
        accuracy = (reward_margin > 0.0).float().mean().item()

    metrics = {
        "dpo_loss": float(loss.item()),
        "accuracy": float(accuracy),
        "reward_margin": float(reward_margin.mean().item()),
        "chosen_reward": float(chosen_implicit_rewards.mean().item()),
        "rejected_reward": float(rejected_implicit_rewards.mean().item())
    }

    return loss, metrics


class DPOModel(nn.Module):
    """
    Wrapper encapsulating both the trainable policy model and frozen reference model.
    """
    def __init__(self, policy_model: MiniGPTChat, reference_model: Optional[MiniGPTChat] = None):
        super().__init__()
        self.policy = policy_model

        if reference_model is None:
            self.reference = copy.deepcopy(policy_model)
        else:
            self.reference = reference_model

        # Strictly freeze reference model
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

    def forward_pairwise(
        self,
        batch: Dict[str, torch.Tensor],
        beta: float = 0.1,
        loss_type: str = "sigmoid"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes policy and reference log probabilities for chosen and rejected responses,
        and evaluates DPO loss.
        """
        # Policy log probabilities (with gradients)
        policy_c_logps = self.policy.get_response_log_probs(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"]
        )
        policy_r_logps = self.policy.get_response_log_probs(
            batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
            labels=batch["rejected_labels"]
        )

        # Reference log probabilities (no gradients)
        with torch.no_grad():
            ref_c_logps = self.reference.get_response_log_probs(
                batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
                labels=batch["chosen_labels"]
            )
            ref_r_logps = self.reference.get_response_log_probs(
                batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
                labels=batch["rejected_labels"]
            )

        loss, metrics = compute_dpo_loss(
            policy_c_logps,
            policy_r_logps,
            ref_c_logps,
            ref_r_logps,
            beta=beta,
            loss_type=loss_type
        )
        return loss, metrics
