"""Test the main application."""

import json
import os
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

from podalize.configs import path2logs, use_auth_token
from podalize.DocumentGenerator import DocumentGenerator
from podalize.myutils import (
    get_diarization,
    get_spoken_time,
    get_transcript,
    get_world_cloud,
    merge_tran_diar,
    mp3wav,
)

if TYPE_CHECKING:
    from pyannote.core.annotation import Annotation


def test_app(sample_audio_two_speakers: str) -> None:
    """Test app on a sample audio."""
    # TODO: split this into different functions, then tests.
    # test diarization
    diarization: Annotation = get_diarization(sample_audio_two_speakers, use_auth_token)
    assert diarization.uri == "sample_audio_two_speakers"
    assert diarization.labels() == ["SPEAKER_00", "SPEAKER_01"]

    # test get_transcript
    raw_transcript = get_transcript(
        model_size="tiny",
        path2audio=mp3wav(sample_audio_two_speakers),
    )
    expected = " This is Rachel for the POTALize Unit tests. Hello Rachel, this is Bill for the POTALize Unit tests."  # noqa: E501
    assert raw_transcript["text"] == expected
    # assert result is None

    # test merge_tran_diar
    p2s = sample_audio_two_speakers.replace(".wav", "_diar.json")
    with Path(p2s).open() as f:
        segements = json.load(f)

    speakers_dict = {}
    for ii, sp in enumerate(diarization.labels()):
        speakers_dict[sp] = f"Speaker_{ii}"

    segements_dict = {}
    for k, v in segements.items():
        k = [float(i) for i in k.split(",")]
        segements_dict[(k[0], k[1])] = v

    transcript = merge_tran_diar(raw_transcript, segements_dict, speakers_dict)
    assert (
        transcript
        == "\n\n\nSpeaker_1           00:00\n This is Rachel for the POTALize Unit tests.\n\n\nSpeaker_0           00:02\n Hello Rachel, this is Bill for the POTALize Unit tests."
    )

    # test get_spoken_time
    spoken_time, spoken_time_secs = get_spoken_time(raw_transcript, speakers_dict)
    print(spoken_time)
    print(spoken_time_secs)
    assert spoken_time == {"SPEAKER_00": "0:00:04", "SPEAKER_01": "0:00:02"}
    assert spoken_time_secs == {"SPEAKER_00": 3.52, "SPEAKER_01": 2.48}

    # test word cloud
    word_cloud = get_world_cloud(raw_transcript, speakers_dict)
    assert word_cloud.sort() == None
    # assert word_cloud.sort() == [
    #     "./data/logs/Speaker_0.png",
    #     "./data/logs/Speaker_1.png",
    # ]

    # test download and DocumentGenerator
    spoken_fig = glob(path2logs + "/spoken*.png")
    all_figs = glob(path2logs + "/*.png")
    wc_figs = [f for f in all_figs if [v for v in speakers_dict.values() if v in f]]

    pod_name = os.path.basename(sample_audio_two_speakers)
    args = {
        "title": pod_name,
        "author": "Created by Podalize",
        "path2logs": path2logs,
    }
    rg = DocumentGenerator(**args)

    for f in spoken_fig:
        rg.add_image(f, caption="Percentage of spoken time per speaker")
        rg.add_new_page()

    for f in wc_figs:
        rg.add_image(f, caption="Word cloud per speaker")
        rg.add_new_page()

    rg.add_section("Transcript", transcript[3:])
    path2pdf = f"{path2logs}/podalize_{pod_name}"
    # rg.doc.generate_pdf(path2pdf, clean_tex=False, compiler='pdfLaTeX')
    # rg.doc.generate_pdf(path2pdf, clean_tex=False)
