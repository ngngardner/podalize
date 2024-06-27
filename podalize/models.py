"""Data models for Podalize app."""

from pathlib import Path

from pydantic import BaseModel, Field


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
