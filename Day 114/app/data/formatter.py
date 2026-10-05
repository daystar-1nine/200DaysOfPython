"""
Chat template formatter for converting multi-role conversation messages
into structured text with delimiter tokens for MiniGPT-Chat.
"""
from typing import List, Dict, Any, Optional


SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


def validate_conversation(messages: List[Dict[str, str]]) -> bool:
    """
    Validates structural integrity of a conversation list:
    - Must be a non-empty list of dicts with 'role' and 'content' keys.
    - Roles must be in {'system', 'user', 'assistant'}.
    - Content must be non-empty string.
    - If 'system' is present, it must be the first message.
    - Must contain at least one 'user' message.
    """
    if not isinstance(messages, list) or len(messages) == 0:
        return False

    valid_roles = {"system", "user", "assistant"}
    has_user = False

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        role = msg["role"]
        content = msg["content"]
        if role not in valid_roles:
            return False
        if not isinstance(content, str) or not content.strip():
            return False
        if role == "system" and idx != 0:
            return False
        if role == "user":
            has_user = True

    return has_user


def format_chat(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False
) -> str:
    """
    Formats structured message history into standard template string:
      <|system|>You are a helpful assistant.<|end|>
      <|user|>What is Python?<|end|>
      <|assistant|>Python is a programming language.<|end|>

    If add_generation_prompt is True, appends '<|assistant|>' to prompt
    the model for its turn during inference.
    """
    formatted_parts: List[str] = []

    for msg in messages:
        role = msg["role"].strip().lower()
        content = msg["content"].strip()

        if role == "system":
            formatted_parts.append(f"{SYSTEM_TOKEN}{content}{END_TOKEN}")
        elif role == "user":
            formatted_parts.append(f"{USER_TOKEN}{content}{END_TOKEN}")
        elif role == "assistant":
            formatted_parts.append(f"{ASSISTANT_TOKEN}{content}{END_TOKEN}")
        else:
            raise ValueError(f"Unsupported message role: '{role}'")

    if add_generation_prompt:
        formatted_parts.append(f"{ASSISTANT_TOKEN}")

    return "".join(formatted_parts)


def extract_assistant_response(raw_generation: str) -> str:
    """
    Extracts the assistant's response from a generated string.
    Finds the last '<|assistant|>' tag and slices until '<|end|>' or EOS.
    """
    if ASSISTANT_TOKEN in raw_generation:
        # Get content after the last assistant token
        after_assistant = raw_generation.split(ASSISTANT_TOKEN)[-1]
    else:
        after_assistant = raw_generation

    # Strip everything after end token if present
    if END_TOKEN in after_assistant:
        clean_text = after_assistant.split(END_TOKEN)[0]
    else:
        clean_text = after_assistant

    return clean_text.strip()
