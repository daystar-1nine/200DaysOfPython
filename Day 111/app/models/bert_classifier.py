"""
BERT SMS Spam Classifier leveraging pretrained Transformer encoder representations.
"""
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from transformers import BertModel
from app.models.classification_head import BertClassificationHead


class BertSpamClassifier(nn.Module):
    """
    BERT Classifier for SMS Spam detection.
    Passes token sequences through pretrained BERT encoder, extracts the [CLS]
    representation from the final layer, and applies a task-specific classification head.
    Supports parameter freezing for transfer learning experiments.
    """
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        num_classes: int = 1,
        dropout_prob: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.bert = BertModel.from_pretrained(model_name, attn_implementation="eager")
        hidden_size = self.bert.config.hidden_size
        self.classifier = BertClassificationHead(hidden_size=hidden_size, num_classes=num_classes, dropout_prob=dropout_prob)

        if freeze_bert:
            self.freeze_backbone(True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freezes or unfreezes all parameters of the underlying BERT encoder."""
        for param in self.bert.parameters():
            param.requires_grad = not freeze

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            output_attentions: whether to return attention matrices
        Returns:
            probs: (batch_size, 1) probability predictions
            cls_repr: (batch_size, hidden_size) [CLS] contextual representations
            attentions: tuple of attention matrices across layers if requested
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        sequence_output = outputs.last_hidden_state  # (B, L, H)
        cls_repr = sequence_output[:, 0, :]          # [CLS] representation (B, H)
        probs = self.classifier(cls_repr)

        attentions = outputs.attentions if output_attentions else None
        return probs, cls_repr, attentions

    def count_parameters(self) -> Dict[str, int]:
        """Returns total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
