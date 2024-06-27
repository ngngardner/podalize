"""Tests for the utils module."""

from pathlib import Path

from podalize.utils import get_audio_format, youtube_downloader


def test_yt_downloader(tmp_path: Path) -> None:
    """Test the youtube downloader."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    audio_record = youtube_downloader(url, tmp_path)
    assert audio_record.file_dir.exists()
    assert audio_record.audio_path.exists()


def test_get_audio_format(tmp_path: Path) -> None:
    """Test the get_audio_format function."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    audio_record = youtube_downloader(url, tmp_path)
    audio_format = get_audio_format(audio_record)
    assert audio_format == "audio/mpeg"
