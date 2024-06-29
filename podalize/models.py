"""Data models for Podalize app."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Segment(BaseModel):
    """Speaker segment model."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    speaker: str | None = None


class Result(BaseModel):
    """Transcription result model."""

    text: str
    segments: list[Segment]
    language: str
    speakers: list[str] | None = None


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
