"""Tests for the utils module."""

import json

from podalize import configs
from podalize.utils import get_audio_format, youtube_downloader


def test_yt_downloader() -> None:
    """Test the youtube downloader."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    if configs.db.get(url, None):
        del configs.db[url]
        with configs.db_path.open("w") as f:
            json.dump(configs.db, f)
    youtube_downloader(url)


def test_get_audio_format() -> None:
    """Test the get_audio_format function."""
    url = "https://www.youtube.com/shorts/nJ4SDNNqBng"
    if configs.db.get(url, None):
        del configs.db[url]
        with configs.db_path.open("w") as f:
            json.dump(configs.db, f)
    audio_record = youtube_downloader(url)
    audio_format = get_audio_format(audio_record)
    assert audio_format == "audio/mpeg"
