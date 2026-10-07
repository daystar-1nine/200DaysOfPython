"""
Generation settings and prompt perturbation options for evaluation runs.
"""
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class EvaluationSettings:
    """Settings controlling model text generation during evaluation."""
    deterministic: bool = True         # True = greedy decoding (temp=0, do_sample=False)
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.9
    max_new_tokens: int = 64
    repetition_penalty: float = 1.1
    seed: int = 42
    stop_tokens: tuple = ("<|end|>", "<|user|>", "<|system|>")

    @property
    def max_tokens(self) -> int:
        return self.max_new_tokens


def apply_prompt_perturbation(prompt: str, mode: str = "clean", seed: int = 42) -> str:
    """
    Applies synthetic perturbations for robustness evaluation:
    - 'clean': returns unmodified prompt
    - 'uppercase': converts entire prompt to UPPERCASE
    - 'whitespace': adds random leading, trailing, and multi-spaces
    - 'typos': introduces minor character swaps/omissions on words
    - 'polite': prepends 'Could you please ' and appends ' thank you'
    """
    if mode == "clean":
        return prompt

    if mode == "uppercase":
        return prompt.upper()

    if mode == "whitespace":
        words = prompt.split()
        return "   " + "   ".join(words) + "   "

    if mode == "polite":
        p = prompt.strip()
        if p.endswith("?"):
            p = p[:-1]
        return f"Could you please {p.lower()}, thank you?"

    if mode == "typos":
        rng = random.Random(seed)
        words = prompt.split()
        perturbed = []
        for w in words:
            if len(w) > 4 and rng.random() < 0.25:
                # Swap two adjacent characters
                idx = rng.randint(1, len(w) - 2)
                w_list = list(w)
                w_list[idx], w_list[idx + 1] = w_list[idx + 1], w_list[idx]
                perturbed.append("".join(w_list))
            else:
                perturbed.append(w)
        return " ".join(perturbed)

    return prompt
