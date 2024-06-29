"""Tests for the utils module."""

from podalize import db
from podalize.utils import get_audio_format, youtube_downloader


def test_yt_downloader() -> None:
    """Test the youtube downloader."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    db.delete_youtube_record(url)
    youtube_downloader(url)
    db.delete_youtube_record(url)


def test_get_audio_format() -> None:
    """Test the get_audio_format function."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    db.delete_youtube_record(url)
    audio_record = youtube_downloader(url)
    audio_format = get_audio_format(audio_record)
    assert audio_format == "audio/mpeg"
    db.delete_youtube_record(url)
