"""
Data validator module for Day 104 Semantic Search Engine.
Validates document schemas, ID uniqueness, and ground-truth consistency.
"""

from typing import Any, Dict, List, Set, Tuple


class DataValidator:
    """Validates schemas and data integrity for documents and ground truth."""

    REQUIRED_DOC_FIELDS = {"id", "title", "text", "category"}
    REQUIRED_QUERY_FIELDS = {"query_id", "query", "relevant_doc_ids"}

    @classmethod
    def validate_documents(cls, docs: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate a list of document dictionaries.
        
        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: List[str] = []
        if not isinstance(docs, list) or len(docs) == 0:
            return False, ["Document collection must be a non-empty list."]

        seen_ids: Set[Any] = set()
        for idx, doc in enumerate(docs):
            if not isinstance(doc, dict):
                errors.append(f"Document at index {idx} is not a dictionary.")
                continue

            missing = cls.REQUIRED_DOC_FIELDS - set(doc.keys())
            if missing:
                errors.append(f"Document at index {idx} is missing required fields: {missing}")

            doc_id = doc.get("id")
            if doc_id is None:
                errors.append(f"Document at index {idx} has None or missing id.")
            elif doc_id in seen_ids:
                errors.append(f"Duplicate document id found: {doc_id} at index {idx}.")
            else:
                seen_ids.add(doc_id)

            for field in ["title", "text", "category"]:
                val = doc.get(field)
                if not isinstance(val, str) or not val.strip():
                    errors.append(f"Document {doc_id} field '{field}' must be a non-empty string.")

        return len(errors) == 0, errors

    @classmethod
    def validate_ground_truth(
        cls, queries: List[Dict[str, Any]], valid_doc_ids: Set[Any] = None
    ) -> Tuple[bool, List[str]]:
        """Validate ground-truth query collection against an optional set of valid doc IDs.
        
        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: List[str] = []
        if not isinstance(queries, list) or len(queries) == 0:
            return False, ["Ground truth collection must be a non-empty list."]

        seen_qids: Set[Any] = set()
        for idx, q in enumerate(queries):
            if not isinstance(q, dict):
                errors.append(f"Query at index {idx} is not a dictionary.")
                continue

            missing = cls.REQUIRED_QUERY_FIELDS - set(q.keys())
            if missing:
                errors.append(f"Query at index {idx} is missing required fields: {missing}")

            qid = q.get("query_id")
            if qid is None:
                errors.append(f"Query at index {idx} has None or missing query_id.")
            elif qid in seen_qids:
                errors.append(f"Duplicate query_id found: {qid} at index {idx}.")
            else:
                seen_qids.add(qid)

            query_text = q.get("query")
            if not isinstance(query_text, str) or not query_text.strip():
                errors.append(f"Query {qid} has empty or non-string query text.")

            rel_ids = q.get("relevant_doc_ids")
            if not isinstance(rel_ids, list):
                errors.append(f"Query {qid} relevant_doc_ids must be a list.")
            elif valid_doc_ids is not None:
                invalid_refs = set(rel_ids) - valid_doc_ids
                if invalid_refs:
                    errors.append(f"Query {qid} references non-existent doc IDs: {invalid_refs}")

        return len(errors) == 0, errors
