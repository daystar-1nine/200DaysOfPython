"""
Context construction and prompt template building for grounded RAG generation.
"""
from typing import List, Tuple, Dict, Any, Optional
from app.retrieval.top_k import QueryResult


DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a truthful, precise, and helpful AI knowledge assistant. "
    "Your objective is to answer the user's question using ONLY the facts provided in the CONTEXT below. "
    "Strictly follow these anti-hallucination instructions:\n"
    "1. If the answer cannot be determined from the provided context, explicitly state: "
    "\"I don't have enough information in the provided documents to answer this question.\"\n"
    "2. Do not invent, extrapolate, or assume facts not directly stated in the context.\n"
    "3. Keep answers concise, factual, and directly relevant to the question."
)


class ContextBuilder:
    """
    Constructs bounded context windows from retrieved QueryResult chunks.
    """
    def __init__(self, max_chars: int = 2500, separator: str = "\n\n"):
        self.max_chars = max_chars
        self.separator = separator

    def build_context(self, results: List[QueryResult]) -> Tuple[str, List[QueryResult]]:
        """
        Formats and bundles retrieved chunks up to max_chars limit.
        Returns:
            (formatted_context_string, list_of_included_query_results)
        """
        if not results:
            return "", []

        context_blocks: List[str] = []
        included_results: List[QueryResult] = []
        accumulated_len = 0

        for r in results:
            meta = r.chunk.metadata
            title = meta.title if meta.title else meta.document_id
            header = f"[Chunk {r.rank} | Source: {title} — ID: {meta.chunk_id}]"
            block = f"{header}\n{r.chunk.text.strip()}"
            block_len = len(block) + len(self.separator)

            if accumulated_len + block_len > self.max_chars and context_blocks:
                break

            context_blocks.append(block)
            included_results.append(r)
            accumulated_len += block_len

        return self.separator.join(context_blocks), included_results


class RAGPromptBuilder:
    """
    Assembles final instruction prompt with system rules, context, and user question.
    """
    def __init__(
        self,
        system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT,
        context_builder: Optional[ContextBuilder] = None
    ):
        self.system_prompt = system_prompt
        self.context_builder = context_builder or ContextBuilder()

    def build_prompt(
        self,
        question: str,
        results: List[QueryResult]
    ) -> Tuple[str, str, List[QueryResult]]:
        """
        Constructs complete RAG prompt.
        Returns:
            (full_prompt, formatted_context, included_results)
        """
        context_str, included = self.context_builder.build_context(results)

        if not context_str.strip():
            context_section = "No relevant context found in documents."
        else:
            context_section = context_str

        prompt = (
            f"SYSTEM:\n{self.system_prompt}\n\n"
            f"CONTEXT:\n{context_section}\n\n"
            f"USER QUESTION:\n{question.strip()}\n\n"
            f"GROUNDED ANSWER:\n"
        )
        return prompt, context_str, included
