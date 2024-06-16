"""Tests for the utils module."""

from pathlib import Path

from podalize.utils import get_audio_format, youtube_downloader


def test_yt_downloader(tmp_path: Path) -> None:
    """Test the youtube downloader."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    audio_path = youtube_downloader(url, tmp_path)
    assert (tmp_path / audio_path.name).exists()


def test_get_audio_format(tmp_path: Path) -> None:
    """Test the get_audio_format function."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    audio_path = youtube_downloader(url, tmp_path)
    audio_format = get_audio_format(audio_path)
    assert audio_format == "audio/mpeg"
