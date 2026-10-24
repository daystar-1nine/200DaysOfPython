"""
Unit tests for document loading, metadata extraction, and text cleaning utilities.
"""
from pathlib import Path
import pytest
from app.ingestion.metadata import DocumentMetadata, ChunkMetadata, create_document_id
from app.ingestion.loader import Document, load_document, load_directory
from app.ingestion.cleaner import clean_text, tokenize_words, extract_sentences, remove_markdown_noise


def test_create_document_id():
    p = Path("python/sub/advanced_generators.md")
    doc_id = create_document_id(p)
    assert doc_id == "doc_advanced_generators"
    assert doc_id.startswith("doc_")


def test_document_metadata_to_dict():
    meta = DocumentMetadata(
        document_id="doc_1",
        title="Test Document",
        category="python",
        source_path="/path/to/test.md",
        file_type=".md",
        char_count=1024,
        word_count=120
    )
    d = meta.to_dict()
    assert d["document_id"] == "doc_1"
    assert d["title"] == "Test Document"
    assert d["category"] == "python"
    assert d["char_count"] == 1024


def test_document_metadata_from_dict():
    data = {
        "document_id": "doc_2",
        "title": "DBMS Notes",
        "category": "dbms",
        "source_path": "/var/data.txt",
        "file_type": ".txt",
        "char_count": 500,
        "word_count": 60
    }
    meta = DocumentMetadata.from_dict(data)
    assert meta.document_id == "doc_2"
    assert meta.category == "dbms"
    assert meta.file_type == ".txt"


def test_clean_text_normalizes_whitespace():
    raw = "  Hello   world! \t This is   a \n\n test.  "
    cleaned = clean_text(raw)
    assert "   " not in cleaned
    assert cleaned.startswith("Hello world!")


def test_clean_text_unicode_normalization():
    raw = "Caf\u00e9\u00a0au\u00a0lait"  # Non-breaking space
    cleaned = clean_text(raw)
    assert "Café au lait" in cleaned


def test_remove_markdown_noise_headers_and_links():
    md = "# Heading 1\n## Subheading\nCheck this [Google](https://google.com) link and **bold** text."
    plain = remove_markdown_noise(md)
    assert "#" not in plain
    assert "https://google.com" not in plain
    assert "Google" in plain
    assert "bold" in plain


def test_tokenize_words_extracts_alphanumeric_tokens():
    text = "RAG retrieval-augmented generation: top_k=5, 100% precision!"
    tokens = tokenize_words(text, lowercase=True)
    assert "rag" in tokens
    assert "retrieval" in tokens
    assert "generation" in tokens
    assert "precision" in tokens


def test_tokenize_words_case_preservation():
    text = "Python NumPy Pandas"
    tokens = tokenize_words(text, lowercase=False)
    assert tokens == ["Python", "NumPy", "Pandas"]


def test_extract_sentences_multiple_terminals():
    text = "Sentence one. Is this two? Yes! And number four..."
    sentences = extract_sentences(text)
    assert len(sentences) >= 3
    assert sentences[0] == "Sentence one."
    assert sentences[1] == "Is this two?"
    assert sentences[2] == "Yes!"


def test_extract_sentences_empty_string():
    assert extract_sentences("") == []
    assert extract_sentences("   \n\t  ") == []


def test_document_serialization():
    doc = Document(
        content="Document content text.",
        metadata=DocumentMetadata(
            document_id="doc_x",
            title="Doc X",
            category="general",
            source_path="x.md",
            file_type=".md",
            char_count=22,
            word_count=3
        )
    )
    d = doc.to_dict()
    assert d["content"] == "Document content text."
    restored = Document.from_dict(d)
    assert restored.metadata.document_id == "doc_x"
    assert restored.content == doc.content


def test_load_document_valid_file(tmp_path):
    sub = tmp_path / "python"
    sub.mkdir()
    file_path = sub / "sample.md"
    file_path.write_text("# Sample Title\n\nThis is document body.", encoding="utf-8")

    doc = load_document(file_path)
    assert doc.metadata.title == "Sample Title"
    assert "document body" in doc.content
    assert doc.metadata.category == "python"
    assert doc.metadata.file_type == ".md"


def test_load_document_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_document(Path("non_existent_file_path_12345.md"))


def test_load_directory_filters_extensions(tmp_path):
    (tmp_path / "doc1.md").write_text("# Doc 1\nContent 1", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Plain text content 2", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01\x02\x03")

    docs = load_directory(tmp_path)
    assert len(docs) == 2
    extensions = {d.metadata.file_type for d in docs}
    assert extensions == {".md", ".txt"}


def test_load_directory_extracts_category_from_parent(tmp_path):
    sub = tmp_path / "machine_learning"
    sub.mkdir()
    (sub / "models.md").write_text("# Models\nContent", encoding="utf-8")

    docs = load_directory(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata.category == "machine_learning"


def test_load_directory_empty_folder(tmp_path):
    docs = load_directory(tmp_path)
    assert docs == []
