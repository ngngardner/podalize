"""Test the main application."""

from typing import TYPE_CHECKING

import pytest

from podalize.configs import use_auth_token
from podalize.models import Record
from podalize.utils import (
    get_diarization,
    get_spoken_time,
    get_transcript,
    get_world_cloud,
    merge_tran_diar,
)
from podalize.webapp.app import (
    handle_document,
    handle_segments,
    handle_speakers,
    process_audio,
)

if TYPE_CHECKING:
    from pyannote.core.annotation import Annotation


def test_diarization(sample_audio_two_speakers: Record) -> None:
    """Test diarization on a sample audio file."""
    diarization: Annotation = get_diarization(sample_audio_two_speakers, use_auth_token)
    assert diarization.uri == "sample_audio_two_speakers"
    assert diarization.labels() == ["SPEAKER_00", "SPEAKER_01"]


def test_get_transcript(
    sample_audio_two_speakers: Record,
    two_speakers_raw_transcript: str,
) -> None:
    """Test transcript generation on a sample audio file."""
    raw_transcript = get_transcript(
        model_size="tiny",
        audio_record=sample_audio_two_speakers,
    )
    assert raw_transcript.text == two_speakers_raw_transcript


def test_handle_speakers(sample_audio_two_speakers: Record) -> None:
    """Test speaker handling on a sample audio file."""
    diarization, labels, y, sr = process_audio(sample_audio_two_speakers)
    speakers = handle_speakers(sample_audio_two_speakers, diarization, labels, y, sr)
    assert speakers == {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"}


def test_transcript(
    sample_audio_two_speakers: Record,
    two_speakers_transcript: str,
) -> None:
    """Test transcript merging on a sample audio file."""
    raw_transcript_json = get_transcript(
        model_size="tiny",
        audio_record=sample_audio_two_speakers,
    )

    diarization, labels, y, sr = process_audio(sample_audio_two_speakers)
    speakers_dict = handle_speakers(
        sample_audio_two_speakers,
        diarization,
        labels,
        y,
        sr,
    )
    segements_dict = handle_segments(sample_audio_two_speakers)
    transcript = merge_tran_diar(raw_transcript_json, segements_dict, speakers_dict)
    assert transcript == two_speakers_transcript

    speakers = list(speakers_dict.keys())
    spoken_time, spoken_time_secs = get_spoken_time(raw_transcript_json, speakers)
    assert spoken_time == {"SPEAKER_00": "0:00:04", "SPEAKER_01": "0:00:02"}
    assert spoken_time_secs == {"SPEAKER_00": 3.52, "SPEAKER_01": 2.48}


@pytest.mark.skip(reason="broken")
def test_word_cloud(sample_audio_two_speakers: Record) -> None:
    """Test world cloud generation on a sample audio file."""
    result = get_transcript(model_size="tiny", audio_record=sample_audio_two_speakers)
    word_cloud = get_world_cloud(
        result,
        {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"},
    )
    assert word_cloud is None


@pytest.mark.skip(reason="requires pdflatex")
def test_handle_document(sample_audio_two_speakers: Record) -> None:
    """Test handle document generation on a sample audio file."""
    diarization, labels, y, sr = process_audio(sample_audio_two_speakers)
    speakers_dict = handle_speakers(
        sample_audio_two_speakers,
        diarization,
        labels,
        y,
        sr,
    )
    pod_name = sample_audio_two_speakers.audio_path.name
    handle_document("placeholder transcript", pod_name, speakers_dict)
