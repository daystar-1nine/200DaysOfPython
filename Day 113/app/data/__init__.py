"""
Data processing package for LLM pretraining pipeline:
loading, cleaning, quality evaluation, deduplication, splitting, and leakage detection.
"""
from app.data.loader import load_raw_text, split_into_documents, build_character_vocab
from app.data.cleaner import normalize_text, quality_score, filter_documents
from app.data.dedup import exact_deduplicate, near_deduplicate, hash_document
from app.data.splitter import train_val_split, detect_leakage, inject_leakage, pack_sequences
