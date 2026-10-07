"""
Preference and chat template formatter for Day 115: Preference Optimization & RLHF.
Converts prompt-chosen-rejected triplets into formatted sequences with special delimiter tokens.
"""
from typing import Dict, Any, Optional, Tuple


SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
PAD_TOKEN = "<|pad|>"

DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate, and concise AI assistant."


def validate_preference_pair(item: Dict[str, Any]) -> bool:
    """
    Validates the structure of a pairwise preference dictionary:
    - Must be a dictionary with 'prompt', 'chosen', and 'rejected' string fields.
    - All three strings must be non-empty after stripping.
    - 'chosen' and 'rejected' must not be identical.
    """
    if not isinstance(item, dict):
        return False

    for key in ("prompt", "chosen", "rejected"):
        if key not in item:
            return False
        val = item[key]
        if not isinstance(val, str) or not val.strip():
            return False

    if item["chosen"].strip() == item["rejected"].strip():
        return False

    return True


def format_prompt(
    prompt_text: str,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True
) -> str:
    """
    Formats a user prompt into standard chat template:
      <|system|>system_prompt<|end|><|user|>prompt_text<|end|><|assistant|>
    """
    sys_part = f"{SYSTEM_TOKEN}{system_prompt.strip()}{END_TOKEN}" if system_prompt else ""
    user_part = f"{USER_TOKEN}{prompt_text.strip()}{END_TOKEN}"
    asst_prefix = f"{ASSISTANT_TOKEN}" if add_generation_prompt else ""
    return f"{sys_part}{user_part}{asst_prefix}"


def format_chat_prompt(
    messages: list,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    add_generation_prompt: bool = True
) -> str:
    """
    Formats a multi-turn chat history into standard delimiter format:
      <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|assistant|>
    """
    parts = []
    if system_prompt:
        parts.append(f"{SYSTEM_TOKEN}{system_prompt.strip()}{END_TOKEN}")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role == "system":
            parts.append(f"{SYSTEM_TOKEN}{content}{END_TOKEN}")
        elif role == "user":
            parts.append(f"{USER_TOKEN}{content}{END_TOKEN}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TOKEN}{content}{END_TOKEN}")

    if add_generation_prompt:
        parts.append(ASSISTANT_TOKEN)

    return "".join(parts)


def format_preference_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
) -> Tuple[str, str, str]:
    """
    Formats a preference pair into:
      (prompt_formatted, chosen_sequence, rejected_sequence)
    where:
      prompt_formatted ends with '<|assistant|>'
      chosen_sequence ends with '<|end|>'
      rejected_sequence ends with '<|end|>'
    """
    prompt_fmt = format_prompt(prompt, system_prompt=system_prompt, add_generation_prompt=True)
    chosen_seq = f"{prompt_fmt}{chosen.strip()}{END_TOKEN}"
    rejected_seq = f"{prompt_fmt}{rejected.strip()}{END_TOKEN}"
    return prompt_fmt, chosen_seq, rejected_seq


def extract_assistant_response(raw_text: str) -> str:
    """
    Extracts the assistant response from a model-generated string:
    Extracts text between the last '<|assistant|>' tag and '<|end|>'.
    """
    if ASSISTANT_TOKEN in raw_text:
        content = raw_text.split(ASSISTANT_TOKEN)[-1]
    else:
        content = raw_text

    if END_TOKEN in content:
        content = content.split(END_TOKEN)[0]

    return content.strip()
