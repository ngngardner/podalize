"""Tests for the utils module."""

from pathlib import Path

from podalize.utils import get_audio_format, youtube_downloader


def test_yt_downloader(tmp_path: Path) -> None:
    """Test the youtube downloader."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    p2audio = youtube_downloader(url, str(tmp_path))
    assert (tmp_path / Path(p2audio).name).exists()


def test_get_audio_format(tmp_path: Path) -> None:
    """Test the get_audio_format function."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    p2audio = youtube_downloader(url, str(tmp_path))
    audio_format = get_audio_format(p2audio)
    assert audio_format == "audio/mpeg"
