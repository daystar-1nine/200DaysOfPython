"""
Unit tests for MetadataFilter, composable predicates, pre-filtering, and post-filtering.
"""
import pytest
from app.retrieval.metadata_filter import MetadataFilter
from app.chunking.chunker import Chunk
from app.index.vector_index import ScoredChunk
from app.ingestion.metadata import ChunkMetadata


def make_chunk(cid: str, category: str, doc_id: str, tags: list, tokens: int = 100) -> Chunk:
    return Chunk(
        text=f"Text content for {cid}",
        metadata=ChunkMetadata(
            chunk_id=cid,
            document_id=doc_id,
            chunk_index=0,
            title="Title",
            category=category,
            start_char=0,
            end_char=50,
            token_count=tokens,
            word_count=tokens,
            tags=tags
        )
    )


@pytest.fixture
def filter_corpus():
    return [
        make_chunk("c1", "python", "doc_py_1", ["syntax", "comprehension"], tokens=80),
        make_chunk("c2", "python", "doc_py_2", ["concurrency", "gil"], tokens=220),
        make_chunk("c3", "dbms", "doc_sql_1", ["sql", "join"], tokens=150),
        make_chunk("c4", "systems", "doc_http_1", ["http", "404"], tokens=90),
        make_chunk("c5", "machine_learning", "doc_ml_1", ["gradient", "optimization"], tokens=300)
    ]


def test_category_predicate(filter_corpus):
    pred = MetadataFilter.build_predicate(category="python")
    filtered = MetadataFilter.pre_filter(filter_corpus, pred)
    assert len(filtered) == 2
    assert all(c.metadata.category == "python" for c in filtered)


def test_multiple_categories_predicate(filter_corpus):
    pred = MetadataFilter.build_predicate(categories=["python", "dbms"])
    filtered = MetadataFilter.pre_filter(filter_corpus, pred)
    assert len(filtered) == 3


def test_document_id_predicate(filter_corpus):
    pred = MetadataFilter.build_predicate(document_id="doc_sql_1")
    filtered = MetadataFilter.pre_filter(filter_corpus, pred)
    assert len(filtered) == 1
    assert filtered[0].metadata.chunk_id == "c3"


def test_tags_predicate_any(filter_corpus):
    pred = MetadataFilter.build_predicate(tags=["gil", "404"], tag_mode="any")
    filtered = MetadataFilter.pre_filter(filter_corpus, pred)
    assert len(filtered) == 2
    ids = {c.metadata.chunk_id for c in filtered}
    assert ids == {"c2", "c4"}


def test_tags_predicate_all(filter_corpus):
    pred = MetadataFilter.build_predicate(tags=["concurrency", "gil"], tag_mode="all")
    filtered = MetadataFilter.pre_filter(filter_corpus, pred)
    assert len(filtered) == 1
    assert filtered[0].metadata.chunk_id == "c2"


def test_token_range_predicate(filter_corpus):
    pred = MetadataFilter.build_predicate(min_tokens=100, max_tokens=250)
    filtered = MetadataFilter.pre_filter(filter_corpus, pred)
    assert len(filtered) == 2
    ids = {c.metadata.chunk_id for c in filtered}
    assert ids == {"c2", "c3"}


def test_custom_predicate_function(filter_corpus):
    pred = MetadataFilter.build_predicate(custom_fn=lambda c: "comprehension" in c.metadata.tags)
    filtered = MetadataFilter.pre_filter(filter_corpus, pred)
    assert len(filtered) == 1
    assert filtered[0].metadata.chunk_id == "c1"


def test_and_filters_combinator(filter_corpus):
    p1 = MetadataFilter.build_predicate(category="python")
    p2 = MetadataFilter.build_predicate(max_tokens=100)
    combined = MetadataFilter.and_filters(p1, p2)
    filtered = MetadataFilter.pre_filter(filter_corpus, combined)
    assert len(filtered) == 1
    assert filtered[0].metadata.chunk_id == "c1"


def test_or_filters_combinator(filter_corpus):
    p1 = MetadataFilter.build_predicate(category="systems")
    p2 = MetadataFilter.build_predicate(category="dbms")
    combined = MetadataFilter.or_filters(p1, p2)
    filtered = MetadataFilter.pre_filter(filter_corpus, combined)
    assert len(filtered) == 2


def test_not_filter_combinator(filter_corpus):
    p1 = MetadataFilter.build_predicate(category="python")
    not_py = MetadataFilter.not_filter(p1)
    filtered = MetadataFilter.pre_filter(filter_corpus, not_py)
    assert len(filtered) == 3
    assert not any(c.metadata.category == "python" for c in filtered)


def test_post_filter(filter_corpus):
    scored = [ScoredChunk(c, 1.0 - i * 0.1) for i, c in enumerate(filter_corpus)]
    pred = MetadataFilter.build_predicate(category="dbms")
    post = MetadataFilter.post_filter(scored, pred)
    assert len(post) == 1
    assert post[0].chunk.metadata.chunk_id == "c3"
