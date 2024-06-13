"""Pytest configuration."""

from pathlib import Path
import pytest
from importlib import resources


# @pytest.fixture(scope="session")
# def tmp_path2audios(tmpdir_factory):
#     return tmpdir_factory.mktemp("data")


@pytest.fixture
def sample_audio_two_speakers() -> str:
    with resources.path("tests.artifacts", "sample_audio_two_speakers.wav") as path:
        return str(path)
