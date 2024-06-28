"""Data models for Podalize app."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from podalize import configs


class Record(BaseModel):
    """Base record for storing scraped, uploaded, and generated data."""

    audio_path: Path = Field(
        ...,
        description="Path to the audio file.",
    )
    file_dir: Path = Field(
        ...,
        description="Directory containing the record.",
    )
    diar_json: Path | None = Field(
        default=None,
        description="Path to the diarization JSON.",
    )
    diar_pkl: Path | None = Field(
        default=None,
        description="Path to the diarization pickle.",
    )
    transcripts: dict[str, Path] = Field(
        default={},
        description="Generated transcripts stored with the model type.",
    )
    speaker_samples: dict[str, Path] = Field(
        default={},
        description="Generated speaker samples.",
    )

    @field_validator("audio_path", "file_dir")
    @classmethod
    def path_must_exist(cls: type["Record"], v: Path) -> Path:
        """Ensure the audio file exists."""
        if not v.exists():
            raise FileNotFoundError
        return v


class YoutubeRecord(Record):
    """Record for storing scraped and generated YouTube data."""

    video_url: str = Field(
        ...,
        description="YouTube video ID.",
    )


def get_youtube_record(url: str) -> YoutubeRecord | None:
    """Read the database for a YouTube fingerprint record."""
    fingerprint = configs.db.get(url, None)
    if fingerprint is None:
        return None
    return YoutubeRecord(
        audio_path=configs.podalize_path / fingerprint / "audio.wav",
        file_dir=configs.podalize_path / fingerprint,
        video_url=url,
    )


def store_youtube_record(youtube_record: YoutubeRecord) -> None:
    """Store a YouTube record in the database."""
    configs.db[youtube_record.video_url] = youtube_record.file_dir.name
    with configs.db_path.open("w") as f:
        json.dump(configs.db, f)
