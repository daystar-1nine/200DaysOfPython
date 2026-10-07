"""
Data collator for Day 115: Preference Optimization & RLHF.
Processes pairwise preference datasets for:
1. Direct Preference Optimization (DPO): tokenizes chosen and rejected sequences with response-only labels.
2. Reward Modeling: batches formatted sequences with attention masks for scalar pooling.
"""
from typing import List, Dict, Any, Optional
import torch
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import format_preference_pair, DEFAULT_SYSTEM_PROMPT


class PreferenceCollator:
    """
    Collates pairwise preference dictionaries into batched PyTorch tensors for DPO and Reward Modeling.
    """
    def __init__(
        self,
        tokenizer: ChatTokenizer,
        max_length: int = 128,
        ignore_index: int = -100,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.system_prompt = system_prompt

    def _tokenize_and_mask(self, prompt_text: str, response_text: str) -> Dict[str, Any]:
        """
        Tokenizes formatted sequence and constructs labels where prompt is masked to ignore_index.
        Only response tokens and final <|end|> token are unmasked.
        """
        full_text = f"{prompt_text}{response_text.strip()}{self.tokenizer.end_token}"
        prompt_ids = self.tokenizer.encode(prompt_text)
        full_ids = self.tokenizer.encode(full_text)

        # Truncate if exceeding max_length
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        prompt_len = min(len(prompt_ids), len(full_ids))

        # Labels: prompt tokens are -100, response tokens are valid token IDs
        labels = [self.ignore_index] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "prompt_length": prompt_len
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Batches a list of preference records.
        Returns:
          {
            "chosen_input_ids": LongTensor [B, T_chosen],
            "chosen_attention_mask": LongTensor [B, T_chosen],
            "chosen_labels": LongTensor [B, T_chosen],
            "rejected_input_ids": LongTensor [B, T_rejected],
            "rejected_attention_mask": LongTensor [B, T_rejected],
            "rejected_labels": LongTensor [B, T_rejected],
            "prompt_lengths": LongTensor [B]
          }
        """
        chosen_records = []
        rejected_records = []

        for item in batch:
            prompt_fmt, chosen_seq, rejected_seq = format_preference_pair(
                item["prompt"], item["chosen"], item["rejected"], system_prompt=self.system_prompt
            )

            c_rec = self._tokenize_and_mask(prompt_fmt, item["chosen"])
            r_rec = self._tokenize_and_mask(prompt_fmt, item["rejected"])

            chosen_records.append(c_rec)
            rejected_records.append(r_rec)

        # Dynamic padding up to max length in this batch (or self.max_length)
        c_max_len = max(len(r["input_ids"]) for r in chosen_records)
        r_max_len = max(len(r["input_ids"]) for r in rejected_records)

        def pad_batch(records: List[Dict[str, Any]], target_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            b_input_ids, b_mask, b_labels = [], [], []
            for r in records:
                cur_len = len(r["input_ids"])
                pad_len = target_len - cur_len

                b_input_ids.append(r["input_ids"] + [self.tokenizer.pad_id] * pad_len)
                b_mask.append([1] * cur_len + [0] * pad_len)
                b_labels.append(r["labels"] + [self.ignore_index] * pad_len)

            return (
                torch.tensor(b_input_ids, dtype=torch.long),
                torch.tensor(b_mask, dtype=torch.long),
                torch.tensor(b_labels, dtype=torch.long)
            )

        from typing import Tuple
        c_ids, c_mask, c_labels = pad_batch(chosen_records, c_max_len)
        r_ids, r_mask, r_labels = pad_batch(rejected_records, r_max_len)

        prompt_lengths = torch.tensor([r["prompt_length"] for r in chosen_records], dtype=torch.long)

        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_labels": c_labels,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_labels": r_labels,
            "prompt_lengths": prompt_lengths
        }
