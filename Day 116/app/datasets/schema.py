"""
Schema definitions and validation for evaluation benchmark datasets.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json


VALID_CATEGORIES = {
    "python", "data_science", "machine_learning", "dbms", "mathematics",
    "general_knowledge", "reasoning", "coding", "instruction_following", "safety"
}

VALID_TYPES = {"text", "code", "json", "boolean", "exact"}


@dataclass
class EvaluationExample:
    """
    Standard schema for a single benchmark evaluation item.
    Supports single or multiple acceptable reference answers, context for factuality,
    executable test cases for code, and structured format constraints.
    """
    id: str
    category: str
    prompt: str
    reference: Optional[str] = None
    references: List[str] = field(default_factory=list)
    context: Optional[str] = None
    expected_type: str = "text"
    supported: Optional[bool] = None
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize references list
        if self.reference and not self.references:
            self.references = [self.reference.strip()]
        elif self.references and not self.reference:
            self.reference = self.references[0].strip()

    def get_references(self) -> List[str]:
        """Returns non-empty list of acceptable reference strings."""
        refs = [r.strip() for r in self.references if r and r.strip()]
        if not refs and self.reference and self.reference.strip():
            refs = [self.reference.strip()]
        return refs

    def validate(self) -> None:
        """
        Validates schema integrity:
        - Non-empty id, prompt
        - Valid category and expected_type
        - At least one reference answer (unless supported is False, where abstention is expected)
        """
        if not self.id or not isinstance(self.id, str) or not self.id.strip():
            raise ValueError(f"Invalid or empty example ID: {self.id}")

        if not self.prompt or not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError(f"Invalid or empty prompt for example ID: {self.id}")

        cat = self.category.lower().strip()
        if cat not in VALID_CATEGORIES:
            raise ValueError(f"Category '{self.category}' not in recognized categories: {VALID_CATEGORIES}")

        if self.expected_type not in VALID_TYPES:
            raise ValueError(f"Type '{self.expected_type}' not in recognized types: {VALID_TYPES}")

        # Unless it's an unsupported hallucination test item, require at least one reference
        if self.supported is not False:
            refs = self.get_references()
            if not refs:
                raise ValueError(f"Example ID '{self.id}' has no valid reference answers.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes example to a JSON-compatible dictionary."""
        d: Dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "reference": self.reference,
            "references": self.get_references(),
            "expected_type": self.expected_type
        }
        if self.context is not None:
            d["context"] = self.context
        if self.supported is not None:
            d["supported"] = self.supported
        if self.test_cases:
            d["test_cases"] = self.test_cases
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationExample":
        """Instantiates an EvaluationExample from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        refs = data.get("references", [])
        ref = data.get("reference", None)
        if isinstance(refs, str):
            refs = [refs]

        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "")),
            prompt=str(data.get("prompt", "")),
            reference=ref,
            references=refs,
            context=data.get("context", None),
            expected_type=data.get("expected_type", "text"),
            supported=data.get("supported", None),
            test_cases=data.get("test_cases", []),
            metadata=data.get("metadata", {})
        )
