"""Unit tests for src/data_loader.py using a tiny synthetic CHNCI-style CSV
(yes — synthetic *fixtures for unit tests*; never used in experiments)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import clean_text, _read_csv_any_encoding  # noqa: E402


@pytest.fixture
def chnci_like_csv(tmp_path):
    df = pd.DataFrame({
        "timestamp": ["2024/9/26 6:02", "2024/9/26 6:12"],
        "platform": ["dy", "wb"],
        "label1": [0, 1], "label2": [0, 1], "label3": [1.0, 1.0],
        "diff": [1, 0],
        "content": ["正常评论 http://t.cn/abc @某人", "垃圾 [流泪]"],
    })
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False, encoding="gb18030")
    return str(path)


def test_read_csv_any_encoding_gb18030(chnci_like_csv):
    df = _read_csv_any_encoding(chnci_like_csv)
    assert list(df.columns) == ["timestamp", "platform", "label1", "label2",
                                "label3", "diff", "content"]
    assert len(df) == 2


def test_clean_text_removes_urls_mentions_keeps_emoticons():
    s = pd.Series(["看 http://t.cn/x 这条 @某人  [流泪]"])
    out = clean_text(s).iloc[0]
    assert "http" not in out and "@某人" not in out and "[流泪]" in out


def test_clean_text_never_drops_rows():
    s = pd.Series([None, "", "  ", "正常"])
    assert len(clean_text(s)) == 4
