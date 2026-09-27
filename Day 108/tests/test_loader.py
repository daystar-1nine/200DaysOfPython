import pytest
from pathlib import Path
import pandas as pd
from app.data.loader import load_raw_data

def test_load_raw_data_valid(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("v1,v2\nham,hello\nspam,free prize\n", encoding="utf-8")
    df = load_raw_data(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["label", "text"]
    assert len(df) == 2
    assert df["label"].iloc[0] == "ham"
    assert df["text"].iloc[1] == "free prize"

def test_load_raw_data_with_label_text_headers(tmp_path):
    csv_file = tmp_path / "test_label_text.csv"
    csv_file.write_text("label,text\nham,meeting tomorrow\n", encoding="utf-8")
    df = load_raw_data(csv_file)
    assert list(df.columns) == ["label", "text"]
    assert len(df) == 1

def test_load_raw_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_sms_file.csv")

def test_load_raw_data_latin1_encoding(tmp_path):
    csv_file = tmp_path / "latin1.csv"
    # Write byte that triggers UnicodeDecodeError under strict utf-8
    csv_file.write_bytes(b"v1,v2\nham,Caf\xe9 and lunch\n")
    df = load_raw_data(csv_file)
    assert len(df) == 1
    assert "Caf" in df["text"].iloc[0]
