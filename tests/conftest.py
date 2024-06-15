"""Pytest configuration."""

from importlib import resources

import pytest
from _pytest.fixtures import SubRequest


@pytest.fixture(
    params=[
        "sample_audio_two_speakers.wav",
        "sample_audio_two_speakers.mp3",
    ],
)
def sample_audio_two_speakers(request: SubRequest) -> str:
    """Sample audio files for unit tests."""
    for ext in ["json", "pkl"]:
        with resources.path(
            "tests.artifacts",
            f"sample_audio_two_speakers_diar.{ext}",
        ) as path:
            if path.exists():
                path.unlink()
        with resources.path(
            "tests.artifacts",
            f"sample_audio_two_speakers_tiny.{ext}",
        ) as path:
            if path.exists():
                path.unlink()
    with resources.path("tests.artifacts", request.param) as path:
        return str(path)


@pytest.fixture()
def two_speakers_raw_transcript() -> str:
    """Sample raw transcript for unit tests."""
    return " This is Rachel for the POTALize Unit tests. Hello Rachel, this is Bill for the POTALize Unit tests."  # noqa: E501


@pytest.fixture()
def two_speakers_transcript() -> str:
    """Sample transcript for unit tests."""
    return "\n\n\nSPEAKER_01           00:00\n This is Rachel for the POTALize Unit tests.\n\n\nSPEAKER_00           00:02\n Hello Rachel, this is Bill for the POTALize Unit tests."  # noqa: E501
