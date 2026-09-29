"""
Challenge 3: Next Sentence Prediction (NSP) input pair generator.
Constructs [CLS] sentence_a [SEP] sentence_b [SEP] with 50/50 IsNext (label=1) and NotNext (label=0) labels.
"""
from typing import List, Dict, Any, Optional
import numpy as np


def generate_nsp_sample(
    sentence_a_ids: List[int],
    sentence_b_ids: List[int],
    is_next: bool,
    cls_token_id: int = 101,
    sep_token_id: int = 102
) -> Dict[str, Any]:
    seg_a = [cls_token_id] + list(sentence_a_ids) + [sep_token_id]
    types_a = [0] * len(seg_a)

    seg_b = list(sentence_b_ids) + [sep_token_id]
    types_b = [1] * len(seg_b)

    input_ids = seg_a + seg_b
    token_type_ids = types_a + types_b
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "token_type_ids": np.array(token_type_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64),
        "label": int(is_next)
    }


if __name__ == "__main__":
    sent_a = [2001, 2002]
    sent_b = [2003, 2004]
    pair = generate_nsp_sample(sent_a, sent_b, is_next=True)
    print("NSP Sample:", pair)
    assert pair["input_ids"][0] == 101  # [CLS]
    assert pair["input_ids"][3] == 102  # [SEP]
    assert pair["input_ids"][-1] == 102 # [SEP]
    assert list(pair["token_type_ids"]) == [0, 0, 0, 0, 1, 1, 1]
    assert pair["label"] == 1
    print("Challenge 3 passed successfully!")
