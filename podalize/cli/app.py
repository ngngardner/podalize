"""Main Podalize cli application module."""

import json
import subprocess

import typer

from podalize import utils
from podalize.logger import get_logger
from podalize.webapp.app import (
    get_youtube_audio,
    handle_speakers,
    process_audio,
)

app = typer.Typer()
logger = get_logger(__name__)


def list_youtube_videos(
    channel_url: str,
) -> list[dict[str, str]]:
    """List all videos from a YouTube channel."""
    cmd = ["yt-dlp", "--flat-playlist", "-J", channel_url]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    playlist = json.loads(result.stdout)
    return [video for channel in playlist["entries"] for video in channel["entries"]]


def transcribe_video(youtube_url: str) -> None:
    """Transcribe a YouTube video."""
    audio_record = get_youtube_audio(youtube_url)
    diarization, labels, y, sr = process_audio(audio_record)
    handle_speakers(audio_record, diarization, labels, y, sr)
    utils.get_transcript(
        model_size="medium",
        audio_record=audio_record,
    )


@app.command()
def main(channel_url: str) -> None:
    """Podalize main command."""
    videos = list_youtube_videos(channel_url)

    logger.info("Videos:")
    for video in videos:
        logger.info("Title: %s", video["title"])
        logger.info("URL: %s", video["url"])
        transcribe_video(video["url"])
