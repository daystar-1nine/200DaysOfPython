"""
Context construction and prompt template building for Advanced RAG generation.
"""
from typing import List, Tuple, Dict, Any, Optional

from app.index.vector_index import ScoredChunk
from app.context.compressor import CompressedContext


DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a truthful, precise, and helpful AI knowledge assistant. "
    "Your objective is to answer the user's question using ONLY the facts provided in the CONTEXT below. "
    "Strictly follow these anti-hallucination instructions:\n"
    "1. If the answer cannot be determined from the provided context, explicitly state: "
    "\"I don't have enough information in the provided documents to answer this question.\"\n"
    "2. Do not invent, extrapolate, or assume facts not directly stated in the context.\n"
    "3. Keep answers concise, factual, and directly cite evidence from the sources."
)


class AdvancedRAGPromptBuilder:
    """
    Builds structured RAG prompts combining system instructions, compressed context, and user questions.
    """
    def __init__(self, system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_prompt(self, question: str, context_text: str) -> str:
        """Constructs full executable prompt."""
        return (
            f"{self.system_prompt}\n\n"
            f"=== CONTEXT START ===\n"
            f"{context_text.strip()}\n"
            f"=== CONTEXT END ===\n\n"
            f"User Question: {question.strip()}\n"
            f"Answer:"
        )
