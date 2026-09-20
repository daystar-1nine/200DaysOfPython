"""
Ground truth manager for Day 104 Semantic Search Engine.
Provides query retrieval and relevance mapping for evaluation.
"""

from typing import Any, Dict, List, Optional, Set


class GroundTruthManager:
    """Manages ground-truth relevance annotations for evaluation benchmark queries."""

    def __init__(self, queries: List[Dict[str, Any]]):
        self.queries = queries
        self._query_map: Dict[str, Dict[str, Any]] = {
            str(q["query_id"]): q for q in queries if "query_id" in q
        }

    def __len__(self) -> int:
        return len(self.queries)

    def get_queries(self) -> List[Dict[str, Any]]:
        """Return all queries."""
        return self.queries

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Return query dict by query ID."""
        return self._query_map.get(str(query_id))

    def get_relevant_ids(self, query_id: str) -> List[Any]:
        """Return list of relevant document IDs for a query."""
        q = self.get_query(query_id)
        if q is None:
            return []
        return q.get("relevant_doc_ids", [])

    def get_queries_by_type(self, query_type: str) -> List[Dict[str, Any]]:
        """Filter queries by query_type (e.g. 'keyword_match', 'semantic_synonym')."""
        return [q for q in self.queries if q.get("query_type") == query_type]

    def get_categories(self) -> Set[str]:
        """Return unique categories across queries."""
        return {q.get("category", "General") for q in self.queries}
