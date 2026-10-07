"""
Interactive Conversational CLI and Side-by-Side Preference Scoring Terminal for Day 115.
Allows live multi-turn dialogue with aligned DPO model, side-by-side comparison against SFT model,
and automated scalar scoring with the Reward Model.
"""
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import torch

from app.config import ModelConfig, RewardModelConfig, GenerationConfig
from app.data.tokenizer import ChatTokenizer
from app.data.formatter import format_prompt, extract_assistant_response, DEFAULT_SYSTEM_PROMPT
from app.models.dpo_model import MiniGPTChat
from app.models.reward_model import RewardModel
from app.training.checkpoint import load_checkpoint


class PreferenceChatSession:
    """
    Manages ongoing conversational state, context window sliding truncation,
    and side-by-side generation scoring with Reward Model.
    """
    def __init__(
        self,
        dpo_model: MiniGPTChat,
        sft_model: Optional[MiniGPTChat] = None,
        reward_model: Optional[RewardModel] = None,
        tokenizer: Optional[ChatTokenizer] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dpo_model = dpo_model.to(self.device).eval()
        self.sft_model = sft_model.to(self.device).eval() if sft_model is not None else None
        self.reward_model = reward_model.to(self.device).eval() if reward_model is not None else None
        self.tokenizer = tokenizer or ChatTokenizer()
        self.system_prompt = system_prompt
        self.gen_cfg = generation_config or GenerationConfig()
        self.max_context = getattr(self.dpo_model.config, "context_length", 128)

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
        always preserving system prompt (index 0).
        """
        limit = max_tokens or self.max_context
        if len(self.messages) <= 1:
            return

        # Iteratively evict oldest user-assistant turn pair (index 1 & 2) if length exceeds limit
        while len(self.messages) > 2:
            formatted = self._format_history_for_prompt()
            if len(self.tokenizer.encode(formatted)) <= limit:
                break
            self.messages.pop(1)
            if len(self.messages) > 1 and self.messages[1]["role"] == "assistant":
                self.messages.pop(1)

    def _format_history_for_prompt(self) -> str:
        parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>{content}<|end|>")
            elif role == "user":
                parts.append(f"<|user|>{content}<|end|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>{content}<|end|>")
        parts.append("<|assistant|>")
        return "".join(parts)

    def generate_reply(self, user_query: str) -> str:
        """
        Appends user query, runs autoregressive generation with DPO model,
        and returns clean assistant response.
        """
        self.add_message("user", user_query)
        self.truncate_history()

        prompt_str = self._format_history_for_prompt()
        p_tokens = self.tokenizer.encode(prompt_str)
        if len(p_tokens) > self.max_context:
            p_tokens = p_tokens[-self.max_context:]
        prompt_ids = torch.tensor([p_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out_ids = self.dpo_model.generate(
                prompt_ids,
                generation_config=self.gen_cfg,
                stop_token_id=self.tokenizer.end_id
            )

        raw_gen = self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)
        reply = extract_assistant_response(raw_gen)
        self.add_message("assistant", reply)
        return reply

    def compare_turn(self, user_query: str) -> Dict[str, Any]:
        """
        Generates responses from both DPO and SFT models, scoring both with Reward Model.
        """
        prompt_str = format_prompt(user_query, system_prompt=self.system_prompt, add_generation_prompt=True)
        p_tokens = self.tokenizer.encode(prompt_str)
        if len(p_tokens) > self.max_context:
            p_tokens = p_tokens[-self.max_context:]
        prompt_ids = torch.tensor([p_tokens], dtype=torch.long, device=self.device)

        # 1. DPO Generation
        with torch.no_grad():
            dpo_out = self.dpo_model.generate(prompt_ids, generation_config=self.gen_cfg, stop_token_id=self.tokenizer.end_id)
        dpo_reply = extract_assistant_response(self.tokenizer.decode(dpo_out[0].tolist(), skip_special_tokens=False))

        # 2. SFT Generation (if available)
        sft_reply = ""
        if self.sft_model is not None:
            with torch.no_grad():
                sft_out = self.sft_model.generate(prompt_ids, generation_config=self.gen_cfg, stop_token_id=self.tokenizer.end_id)
            sft_reply = extract_assistant_response(self.tokenizer.decode(sft_out[0].tolist(), skip_special_tokens=False))

        # 3. Reward Model Scoring (if available)
        dpo_reward = None
        sft_reward = None
        if self.reward_model is not None:
            rm_ctx = getattr(self.reward_model.config, "context_length", self.max_context)
            def score_text(resp_text: str) -> float:
                full_seq = f"{prompt_str}{resp_text}<|end|>"
                seq_tokens = self.tokenizer.encode(full_seq)
                if len(seq_tokens) > rm_ctx:
                    seq_tokens = seq_tokens[:rm_ctx]
                t_ids = torch.tensor([seq_tokens], dtype=torch.long, device=self.device)
                t_mask = torch.ones_like(t_ids)
                with torch.no_grad():
                    rew = self.reward_model(t_ids, attention_mask=t_mask)
                return float(rew.item())

            dpo_reward = round(score_text(dpo_reply), 3)
            if sft_reply:
                sft_reward = round(score_text(sft_reply), 3)

        return {
            "query": user_query,
            "dpo_response": dpo_reply,
            "dpo_reward": dpo_reward,
            "sft_response": sft_reply,
            "sft_reward": sft_reward
        }


def run_interactive_cli():
    """Runs interactive terminal for preference testing."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    checkpoints_dir = base_dir / "outputs" / "checkpoints"
    data_dir = base_dir / "data"

    tokenizer = ChatTokenizer.load(data_dir / "vocab.json")
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size)

    dpo_model = MiniGPTChat(model_cfg)
    dpo_ckpt = checkpoints_dir / "dpo_training_metrics" / "best_model.pt"
    if dpo_ckpt.exists():
        load_checkpoint(dpo_ckpt, dpo_model)

    rm_model = RewardModel(model_config=model_cfg)
    rm_ckpt = checkpoints_dir / "reward_model_metrics" / "best_model.pt"
    if rm_ckpt.exists():
        load_checkpoint(rm_ckpt, rm_model)

    session = PreferenceChatSession(
        dpo_model=dpo_model,
        reward_model=rm_model,
        tokenizer=tokenizer
    )

    print("=" * 80)
    print("MINIGPT PREFERENCE LAB: INTERACTIVE TERMINAL")
    print("=" * 80)
    print("Commands: /reset, /compare <query>, /temp <val>, /exit\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() == "/exit":
            break
        elif user_input.lower() == "/reset":
            session.reset()
            print("[System] Conversation history cleared.\n")
            continue
        elif user_input.startswith("/compare "):
            query = user_input[9:].strip()
            res = session.compare_turn(query)
            print(f"\n[DPO Aligned Response] (Reward: {res['dpo_reward']}):\n{res['dpo_response']}\n")
            continue

        reply = session.generate_reply(user_input)
        print(f"MiniGPT-DPO: {reply}\n")


if __name__ == "__main__":
    run_interactive_cli()
