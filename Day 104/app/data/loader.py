"""
Data loader module for Day 104 Semantic Search Engine.
Handles loading raw documents, ground-truth queries, and processed corpora.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


class DataLoader:
    """Provides methods for reading and writing document datasets and evaluation benchmarks."""

    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Any:
        """Load data from a JSON file.
        
        Args:
            filepath: Path to the JSON file.
            
        Returns:
            Parsed JSON content.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid JSON.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")

    @classmethod
    def load_documents(cls, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load document collection from JSON.
        
        Args:
            filepath: Path to documents.json.
            
        Returns:
            List of document dictionaries with keys: id, title, text, category.
        """
        data = cls.load_json(filepath)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of documents, got {type(data).__name__}")
        return data

    @classmethod
    def load_ground_truth(cls, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load ground-truth query collection from JSON.
        
        Args:
            filepath: Path to ground_truth.json.
            
        Returns:
            List of query dictionaries with keys: query_id, query, relevant_doc_ids, etc.
        """
        data = cls.load_json(filepath)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of queries, got {type(data).__name__}")
        return data

    @staticmethod
    def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
        """Save data to a JSON file.
        
        Args:
            data: Data to serialize.
            filepath: Destination file path.
            indent: Indentation level for pretty printing.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
