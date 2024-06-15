"""Podalize utilities module."""

import datetime
import hashlib
import json
import os
import pickle
import shutil
import time
import uuid
from pathlib import Path
from textwrap import dedent

import magic
import matplotlib.pyplot as plt
import streamlit as st
import whisper
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
from pydantic import BaseModel
from wordcloud import WordCloud

from podalize.logger import get_logger

logger = get_logger(__name__)


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


def hash_audio_file(file_path: Path, chunk_size: int = 8192) -> str:
    """Hash an audio file to create a unique reusable identifier."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as audio_file:
        while chunk := audio_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def youtube_downloader(url: str, destination: Path) -> Path:
    """Download a youtube video to a destination folder."""
    path2mp3 = destination / f"audio_{uuid.uuid4()}.mp3"
    os.system(f"yt-dlp -x --audio-format mp3 -o {path2mp3} {url}")  # noqa: S605

    ident = hash_audio_file(path2mp3)
    out_path = destination / f"audio_{ident}.mp3"
    shutil.move(path2mp3, out_path)
    return out_path


def merge_tran_diar(  # noqa: C901
    result: Result,
    segements_dict: dict[tuple[float, float], str],
    speakers_dict: dict[str, str],
) -> str:
    """Merge the transcription and diarization results."""
    output = ""
    prev_sp = ""
    transcribed: set[int] = set()
    for idx, seg in enumerate(result.segments):
        if idx in transcribed:
            continue
        start = str(datetime.timedelta(seconds=round(seg.start, 0)))
        if start.startswith("0"):
            start = start[2:]
        end = str(datetime.timedelta(seconds=round(seg.end, 0)))
        if end.startswith("0"):
            end = end[2:]

        overlaps = {}
        for k, sp in segements_dict.items():
            if seg.start > k[1] or seg.end < k[0]:
                continue

            ov = get_overlap(k, (seg.start, seg.end))
            if ov >= 0.3:  # noqa: PLR2004
                overlaps[sp] = ov
                transcribed.add(idx)
        if overlaps:
            sp = max(overlaps, key=lambda x: overlaps[x])
            result.segments[idx].speaker = sp
            ov = max(overlaps.values())
            if sp != prev_sp:
                out = "\n\n\n" + f"{sp}           {start}\n" + seg.text
                output += out
                prev_sp = sp
            else:
                prev_sp = sp
                out = seg.text
                output += out
    result.speakers = list(set(segements_dict.values()))

    for sp in speakers_dict:
        output = output.replace(sp, speakers_dict[sp])
    return output


def get_segments(
    diarization: Annotation,
    speaker_dict: dict[str, str],
) -> dict[str, str]:
    """Get the segments from the diarization."""
    segments_dict = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_end = f"{turn.start},{turn.end}"
        segments_dict[start_end] = speaker_dict.get(speaker, speaker)
    return segments_dict


def get_largest_duration(
    diarization: Annotation,
    speaker: str,
    max_length: float = 5.0,
) -> tuple[int, int, float]:
    """Get the longest duration for a speaker."""
    maxsofar = -float("Inf")
    s, e, d = 0, 0, 0
    for turn, _, sp in diarization.itertracks(yield_label=True):
        d = turn.end - turn.start
        if d > maxsofar and sp == speaker:
            maxsofar = d
            s, e = turn.start, turn.end
    logger.debug(
        dedent(f"""
        {speaker},
        start: {s:.1f}
        end: {e:.1f}
        duration: {maxsofar:.1f} secs
        """),
    )
    m = (s + e) / 2
    s = int(max(0, m - max_length / 2))
    e = int(min(m + max_length / 2, e))
    return s, e, maxsofar


def get_transcript(model_size: str, path2audio: Path) -> Result:
    """Get the transcript for an audio file from a model."""
    # check if trainscript available
    p2f = path2audio.with_name(f"{path2audio.stem}_{model_size}.json")
    if p2f.exists():
        with p2f.open("r") as f:
            result = json.load(f)
        return Result(**result)

    logger.debug("loading model ...")
    model = whisper.load_model(model_size)

    logger.debug("transcribe ...")
    st = time.time()
    result = model.transcribe(str(path2audio))
    el = time.time() - st
    logger.debug("elapsed time: %s", str(datetime.timedelta(seconds=el)))

    # store transcript
    with p2f.open("w") as f:
        json.dump(result, f)

    return Result(**result)


def get_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Get the overlap between two segments."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def audio2wav(p2audio: Path) -> Path:
    """Convert an audio file to wav."""
    if p2audio.suffix == ".wav":
        logger.debug("audio is in wav format!")
        return p2audio
    p2wav = p2audio.with_name(f"{p2audio.stem}.wav")
    if p2wav.exists():
        logger.debug("%s exists!", p2audio)
        return p2wav

    from pydub import AudioSegment

    logger.debug("loading %s", p2audio)
    sound = AudioSegment.from_file(p2audio)

    logger.debug("exporting to % s", p2wav)
    sound.export(p2wav, format="wav")
    return p2wav


def get_diarization(p2audio: Path, use_auth_token: str) -> Annotation:
    """Get the diarization from the audio file."""
    p2s = p2audio.with_name(f"{p2audio.stem}_diar.json")
    p2p = p2audio.with_name(f"{p2audio.stem}_diar.pkl")
    if p2p.exists():
        logger.debug("loading %s", p2p)
        with p2p.open("rb") as handle:
            diarization = pickle.load(handle)  # noqa: S301

        speaker_dict: dict[str, str] = {}
        segments_dict = get_segments(diarization, speaker_dict)
        with p2s.open("w") as f:
            json.dump(segments_dict, f)
            logger.debug("dumped diarization to %s", p2s)

    else:
        p2audio = audio2wav(p2audio)
        logger.debug("loading model ...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=use_auth_token,
        )

        logger.debug("diarization ...")
        diarization = pipeline(p2audio)

        # save diarization
        with p2s.open("wb") as handle:
            pickle.dump(diarization, handle, protocol=pickle.HIGHEST_PROTOCOL)

        speaker_dict = {}
        segments_dict = get_segments(diarization, speaker_dict)
        with p2s.open("w") as f:
            json.dump(segments_dict, f)
            logger.debug("dumped diarization to %s", p2s)

    return diarization


def get_spoken_time(
    result: Result,
    speakers: list[str],
) -> tuple[dict[str, str], dict[str, float]]:
    """Get spoken time for each speaker."""
    spoken_time = {}
    spoken_time_secs = {}
    for sp in speakers:
        st = sum(
            [seg.end - seg.start for seg in result.segments if seg.speaker == sp],
        )
        spoken_time_secs[sp] = st
        spoken_time[sp] = str(datetime.timedelta(seconds=round(st, 0)))
    return spoken_time, spoken_time_secs


def get_world_cloud(
    result: Result,
    speakers_dict: dict[str, str],
    path2figs: str = "./data/logs",
) -> list[str]:
    """Generate a wordcloud figure for each speaker."""
    figs: list[str] = []
    if result.speakers is None:
        return figs

    for sp in result.speakers:
        words = "".join(
            [seg.text for seg in result.segments if seg.speaker == sp],
        )
        if words == "":
            continue
        wordcloud = WordCloud(max_font_size=40, background_color="white").generate(
            words,
        )
        fig = plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(speakers_dict[sp])
        plt.axis("off")
        st.pyplot(fig)

        p2f = f"{path2figs}/{speakers_dict[sp]}.png"
        figs.append(p2f)
        plt.savefig(p2f)
    return figs


def get_audio_format(p2a: Path) -> str:
    """Given a path to an audio file, return the file format."""
    with p2a.open("rb") as f:
        audio_data = f.read()
    audio_format = magic.from_buffer(audio_data, mime=True)
    logger.debug("The audio file is in the format %s.", audio_format)
    return audio_format
