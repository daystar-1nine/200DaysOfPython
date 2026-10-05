"""
Interactive chat inference interface and conversation memory manager for MiniGPT-Chat.
Supports multi-turn dialog, dynamic context window truncation, and generation parameter control.
"""
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import torch
from app.config import ChatModelConfig, GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import format_chat, extract_assistant_response
from app.model.minigpt_chat import MiniGPTChat
from app.training.checkpoint import load_sft_checkpoint


class ChatSession:
    """
    Manages ongoing multi-turn chat session state, message truncation,
    and autoregressive response generation.
    """
    def __init__(
        self,
        model: MiniGPTChat,
        tokenizer: ChatTokenizer,
        system_prompt: str = "You are a helpful, accurate, and concise AI assistant.",
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.gen_config = generation_config or GenerationConfig()
        self.max_context = getattr(self.model.config, "context_length", 128)

        self.messages: List[Dict[str, str]] = []
        self.reset()

    def reset(self) -> None:
        """Clears dialog history and reinitializes with system prompt."""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def add_message(self, role: str, content: str) -> None:
        """Appends a message to the conversation history."""
        self.messages.append({"role": role, "content": content.strip()})

    def truncate_history(self, max_tokens: Optional[int] = None) -> None:
        """
        Truncates conversation history if total tokens exceed max_tokens,
        always preserving the system prompt (index 0).
        """
        limit = max_tokens or self.max_context
        # Never truncate if only system prompt is present
        if len(self.messages) <= 1:
            return

        formatted = format_chat(self.messages, add_generation_prompt=True)
        tokens = self.tokenizer.encode(formatted)

        while len(tokens) > (limit - self.gen_config.max_new_tokens) and len(self.messages) > 2:
            # Drop the oldest user-assistant exchange (indices 1 and 2)
            self.messages.pop(1)
            formatted = format_chat(self.messages, add_generation_prompt=True)
            tokens = self.tokenizer.encode(formatted)

    def respond(
        self,
        user_input: str,
        override_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Takes user input, formats history, executes generation, and returns assistant reply.
        """
        config = override_config or self.gen_config
        self.add_message("user", user_input)
        self.truncate_history()

        formatted_prompt = format_chat(self.messages, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(formatted_prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generated_ids = self.model.generate(
            prompt_ids=prompt_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            stop_token_id=self.tokenizer.end_id,
            do_sample=config.do_sample
        )

        full_output = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
        reply = extract_assistant_response(full_output)

        self.add_message("assistant", reply)
        return reply


def run_interactive_cli(
    checkpoint_path: Optional[Union[str, Path]] = None,
    vocab_path: Optional[Union[str, Path]] = None
) -> None:
    """Runs interactive terminal chat loop."""
    print("=" * 60)
    print("MiniGPT-Chat Interactive Console (Day 114)")
    print("Commands: /clear, /history, /temp <val>, /exit")
    print("=" * 60)

    vocab_p = Path(vocab_path or "data/vocab.json")
    if vocab_p.exists():
        tokenizer = ChatTokenizer.load(vocab_p)
    else:
        tokenizer = ChatTokenizer()

    cfg = ChatModelConfig(vocab_size=tokenizer.vocab_size, context_length=128, embed_dim=128, num_heads=4, num_layers=4)
    model = MiniGPTChat(cfg)

    if checkpoint_path and Path(checkpoint_path).exists():
        load_sft_checkpoint(checkpoint_path, model)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print("Using randomly initialized base model (no checkpoint provided).")

    session = ChatSession(model, tokenizer)

    while True:
        try:
            user_text = input("\nYou: ").strip()
            if not user_text:
                continue
            if user_text.lower() == "/exit":
                print("Exiting chat.")
                break
            elif user_text.lower() == "/clear":
                session.reset()
                print("[Chat memory cleared]")
                continue
            elif user_text.lower() == "/history":
                for msg in session.messages:
                    print(f"[{msg['role']}]: {msg['content']}")
                continue
            elif user_text.lower().startswith("/temp "):
                try:
                    val = float(user_text.split()[1])
                    session.gen_config.temperature = val
                    print(f"[Temperature set to {val}]")
                except ValueError:
                    print("[Invalid temperature]")
                continue

            reply = session.respond(user_text)
            print(f"Assistant: {reply}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    run_interactive_cli()
