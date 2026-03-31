"""Tests for Sentinel-2 filename → image acquisition time display."""

from __future__ import annotations

import unittest

from vessel_detection.s2_download import image_acquisition_display_utc_from_tci_filename


class TestImageAcquisitionFromFilename(unittest.TestCase):
    def test_l2a_tci_10m_name(self) -> None:
        name = (
            "S2A_MSIL2A_20240615T103031_N0512_R065_T32TQR_20240615T123456"
            "_TCI_10m.jp2"
        )
        s = image_acquisition_display_utc_from_tci_filename(name)
        self.assertEqual(s, "2024-06-15, 10:30:31 UTC")

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(image_acquisition_display_utc_from_tci_filename("foo.tif"))


if __name__ == "__main__":
    unittest.main()
