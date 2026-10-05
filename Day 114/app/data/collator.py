"""
Data collator for Supervised Fine-Tuning (SFT) with dynamic padding,
attention masking, and prompt loss masking.
"""
from typing import List, Dict, Any, Optional
import torch
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import format_chat


class SFTDataCollator:
    """
    Batches conversation dictionaries, tokenizes them, pads sequences,
    generates causal attention masks, and applies loss masking so that
    training loss is computed ONLY on assistant responses.
    """
    def __init__(
        self,
        tokenizer: ChatTokenizer,
        max_length: int = 128,
        mask_prompt_loss: bool = True,
        ignore_index: int = -100
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt_loss = mask_prompt_loss
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of conversation records.
        Returns:
          {
            "input_ids": LongTensor [B, T],
            "attention_mask": LongTensor [B, T],
            "labels": LongTensor [B, T]
          }
        """
        all_input_ids: List[List[int]] = []
        all_attention_masks: List[List[int]] = []
        all_labels: List[List[int]] = []

        for item in batch:
            messages = item["messages"]
            formatted_text = format_chat(messages, add_generation_prompt=False)
            token_ids = self.tokenizer.encode(formatted_text)

            # Truncate if exceeding max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]

            # Construct labels with assistant loss masking
            if self.mask_prompt_loss:
                labels = [self.ignore_index] * len(token_ids)
                # Find all occurrences of assistant blocks: between assistant_id and end_id
                in_assistant = False
                for idx, tid in enumerate(token_ids):
                    if tid == self.tokenizer.assistant_id:
                        in_assistant = True
                        labels[idx] = self.ignore_index  # Do not train on the <|assistant|> tag itself
                    elif in_assistant:
                        labels[idx] = tid  # Train on assistant response content and <|end|>
                        if tid == self.tokenizer.end_id:
                            in_assistant = False
            else:
                # Standard causal language modeling (predict all non-padding tokens)
                labels = list(token_ids)

            # Padding
            seq_len = len(token_ids)
            pad_len = self.max_length - seq_len

            padded_input_ids = token_ids + [self.tokenizer.pad_id] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
            padded_labels = labels + [self.ignore_index] * pad_len

            all_input_ids.append(padded_input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long)
        }
