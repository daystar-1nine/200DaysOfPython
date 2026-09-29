"""
Next Sentence Prediction (NSP) input pair generation from scratch.
Constructs [CLS] A [SEP] B [SEP] inputs with segment token type IDs.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def create_nsp_pair(
    sentence_a_ids: List[int],
    sentence_b_ids: List[int],
    is_next: bool,
    cls_token_id: int = 101,
    sep_token_id: int = 102,
    pad_token_id: int = 0,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Constructs a dual-sentence input sequence according to the original BERT format:
        [CLS] sentence_a [SEP] sentence_b [SEP] [PAD] ...

    Args:
        sentence_a_ids: token IDs for segment A
        sentence_b_ids: token IDs for segment B
        is_next: True if sentence B follows sentence A (label=1), False otherwise (label=0)
        cls_token_id: [CLS] ID
        sep_token_id: [SEP] ID
        pad_token_id: [PAD] ID
        max_length: optional maximum sequence length with padding/truncation
    Returns:
        dict containing:
            - input_ids: combined token IDs
            - token_type_ids: 0 for segment A and special tokens, 1 for segment B
            - attention_mask: 1 for real tokens, 0 for padding
            - label: 1 if is_next else 0
    """
    # Truncate if necessary (accounting for [CLS], [SEP], [SEP])
    if max_length is not None:
        max_tokens = max_length - 3
        total_len = len(sentence_a_ids) + len(sentence_b_ids)
        if total_len > max_tokens:
            # Truncate the longer sequence first
            excess = total_len - max_tokens
            if len(sentence_a_ids) > len(sentence_b_ids):
                sentence_a_ids = sentence_a_ids[:-excess]
            else:
                sentence_b_ids = sentence_b_ids[:-excess]

    # Build sequence
    # [CLS] A [SEP] -> segment 0
    seg_a = [cls_token_id] + list(sentence_a_ids) + [sep_token_id]
    types_a = [0] * len(seg_a)

    # B [SEP] -> segment 1
    seg_b = list(sentence_b_ids) + [sep_token_id]
    types_b = [1] * len(seg_b)

    input_ids = seg_a + seg_b
    token_type_ids = types_a + types_b
    attention_mask = [1] * len(input_ids)

    # Pad if max_length is specified
    if max_length is not None and len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids += [pad_token_id] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "token_type_ids": np.array(token_type_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64),
        "label": int(is_next)
    }


if __name__ == "__main__":
    sent_a = [2023, 2003, 1037, 3231]  # "this is a test"
    sent_b = [2009, 2003, 2307]        # "it is great"
    pair = create_nsp_pair(sent_a, sent_b, is_next=True, max_length=15)
    print("NSP Input IDs:      ", pair["input_ids"])
    print("NSP Token Type IDs: ", pair["token_type_ids"])
    print("NSP Attention Mask: ", pair["attention_mask"])
    print("NSP Label:          ", pair["label"])
    assert len(pair["input_ids"]) == 15
    assert pair["token_type_ids"][0] == 0
    assert pair["token_type_ids"][-1] == 0  # pad
    assert pair["label"] == 1
    print("NSP pair generation verified!")
