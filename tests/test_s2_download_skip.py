"""Tests for skip-if-on-disk download behavior (S3 quota)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aquaforge.s2_download import (
    TciSclDownloadOutcome,
    download_item_asset,
    tci_scl_download_summary,
)


class TestSkipExisting(unittest.TestCase):
    def test_skips_s3_when_file_nonempty(self) -> None:
        item = {
            "id": "S2A_TEST",
            "assets": {"TCI_10m": {"href": "s3://bucket/prefix/TCI.jp2"}},
        }
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            dest = d / "S2A_TEST_TCI.jp2"
            dest.write_bytes(b"x")
            with patch("aquaforge.s2_download.download_s3_asset") as mock_s3:
                p, skipped = download_item_asset(item, "TCI_10m", d, "token")
                self.assertTrue(skipped)
                self.assertEqual(p, dest)
                mock_s3.assert_not_called()

    @patch.dict(
        "os.environ",
        {"COPERNICUS_S3_ACCESS_KEY": "a", "COPERNICUS_S3_SECRET_KEY": "b"},
        clear=False,
    )
    @patch("aquaforge.s2_download.download_s3_asset")
    def test_downloads_when_missing(self, mock_s3: object) -> None:
        item = {
            "id": "S2A_TEST",
            "assets": {"TCI_10m": {"href": "s3://bucket/prefix/TCI.jp2"}},
        }
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            p, skipped = download_item_asset(item, "TCI_10m", d, "token")
            self.assertFalse(skipped)
            self.assertTrue(mock_s3.called)


class TestSummary(unittest.TestCase):
    def test_both_skipped(self) -> None:
        p = Path("/x/TCI.jp2")
        s = tci_scl_download_summary(TciSclDownloadOutcome(p, p, True, True))
        self.assertIn("skipped", s.lower())

    def test_no_scl(self) -> None:
        p = Path("/x/TCI.jp2")
        s = tci_scl_download_summary(TciSclDownloadOutcome(p, None, False, False))
        self.assertIn("not listed", s.lower())


if __name__ == "__main__":
    unittest.main()
