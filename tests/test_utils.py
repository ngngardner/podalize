"""Tests for the utils module."""

import os
from pathlib import Path

from podalize.utils import get_audio_format, youtube_downloader


def test_yt_downloader(tmp_path: Path):
    """Test the youtube downloader."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    p2audio = youtube_downloader(url, tmp_path)
    assert os.path.exists(tmp_path / Path(p2audio).name)


def test_get_audio_format(tmp_path: Path):
    """Test the get_audio_format function."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    p2audio = youtube_downloader(url, tmp_path)
    audio_format = get_audio_format(p2audio)
    assert audio_format == "audio/mpeg"
