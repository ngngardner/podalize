"""Pytest configuration."""

from importlib import resources

import pytest

# @pytest.fixture(scope="session")
# def tmp_path2audios(tmpdir_factory):
#     return tmpdir_factory.mktemp("data")


@pytest.fixture()
def sample_audio_two_speakers() -> str:
    for ext in ["json", "pkl"]:
        with resources.path(
            "tests.artifacts", f"sample_audio_two_speakers_diar.{ext}",
        ) as path:
            if path.exists():
                path.unlink()
        with resources.path(
            "tests.artifacts", f"sample_audio_two_speakers_tiny.{ext}",
        ) as path:
            if path.exists():
                path.unlink()

    with resources.path("tests.artifacts", "sample_audio_two_speakers.wav") as path:
        return str(path)


@pytest.fixture()
def two_speakers_raw_transcript() -> str:
    return " This is Rachel for the POTALize Unit tests. Hello Rachel, this is Bill for the POTALize Unit tests."  # noqa: E501


@pytest.fixture()
def two_speakers_transcript() -> str:
    return "\n\n\nSPEAKER_01           00:00\n This is Rachel for the POTALize Unit tests.\n\n\nSPEAKER_00           00:02\n Hello Rachel, this is Bill for the POTALize Unit tests."  # noqa: E501
