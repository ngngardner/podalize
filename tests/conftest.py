"""Pytest configuration."""

import shutil
from importlib import resources
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest

import podalize
from podalize.models import Record
from podalize.utils import audio_fingerprint_dir


@pytest.fixture(autouse=True)
def _setup_dirs(tmp_path: Path) -> None:
    """Initialize directories."""
    podalize_path = tmp_path / ".podalize"
    log_path = podalize_path / "logs"
    podalize_path.mkdir()
    log_path.mkdir()
    podalize.configs.podalize_path = podalize_path
    podalize.configs.log_path = log_path


@pytest.fixture(
    params=[
        "sample_audio_two_speakers.wav",
        "sample_audio_two_speakers.mp3",
    ],
)
def sample_audio_two_speakers(request: SubRequest) -> Record:
    """Sample audio files for unit tests."""
    with resources.path("tests.artifacts", request.param) as path:
        dest: Path = podalize.configs.podalize_path / request.param
        shutil.copy(path, dest)
    audio_record = Record(audio_path=dest, file_dir=dest.parent)
    audio_fingerprint_dir(audio_record)
    return audio_record


@pytest.fixture()
def two_speakers_raw_transcript() -> str:
    """Sample raw transcript for unit tests."""
    return " This is Rachel for the POTALize Unit tests. Hello Rachel, this is Bill for the POTALize Unit tests."  # noqa: E501


@pytest.fixture()
def two_speakers_transcript() -> str:
    """Sample transcript for unit tests."""
    return "\n\n\nSPEAKER_01           00:00\n This is Rachel for the POTALize Unit tests.\n\n\nSPEAKER_00           00:02\n Hello Rachel, this is Bill for the POTALize Unit tests."  # noqa: E501
