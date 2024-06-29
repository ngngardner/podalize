"""Podalize utilities module."""

import datetime
import hashlib
import json
import os
import pickle
import shutil
import time
import uuid
from textwrap import dedent

import magic
import matplotlib.pyplot as plt
import streamlit as st
import whisper
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
from pydub import AudioSegment
from wordcloud import WordCloud

from podalize import configs, db, models
from podalize.logger import get_logger

logger = get_logger(__name__)


def hash_audio_file(audio_record: models.Record, chunk_size: int = 8192) -> str:
    """Hash an audio file to create a unique reusable identifier."""
    hasher = hashlib.sha256()
    if audio_record.audio_path.exists():
        with audio_record.audio_path.open("rb") as audio_file:
            while chunk := audio_file.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    raise FileNotFoundError


def audio_fingerprint_dir(audio_record: models.Record) -> None:
    """Create the fingerprint dir for a given audio file."""
    fingerprint = hash_audio_file(audio_record)
    dest = configs.podalize_path / fingerprint
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(audio_record.audio_path, dest / audio_record.audio_path.name)

    audio_record.file_dir = dest
    audio_record.audio_path = dest / audio_record.audio_path.name


def youtube_downloader(url: str) -> models.YoutubeRecord:
    """Download a youtube video to a destination folder."""
    if audio_record := db.get_youtube_record(url):
        return audio_record
    mp3_path = configs.tmp_path / f"audio_{uuid.uuid4()}.mp3"
    command = f"yt-dlp -x --audio-format mp3 -o {mp3_path} {url}"
    os.system(command)  # noqa: S605
    retry_count = 0
    while not mp3_path.exists():
        # retry
        time.sleep(1)
        os.system(command)  # noqa: S605
        retry_count += 1
        if retry_count > 3:  # noqa: PLR2004
            msg = "Failed to download youtube video"
            raise RuntimeError(msg)

    out_path = configs.tmp_path / "audio.mp3"
    shutil.move(mp3_path, out_path)
    audio_record = models.YoutubeRecord(
        video_url=url,
        audio_path=out_path,
        file_dir=configs.tmp_path,
    )
    audio_fingerprint_dir(audio_record)
    db.store_youtube_record(audio_record)
    return audio_record


def merge_tran_diar(  # noqa: C901
    result: models.Result,
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


def get_transcript(model_size: str, audio_record: models.Record) -> models.Result:
    """Get the transcript for an audio file from a model."""
    # check if trainscript available
    audio_record.transcripts[model_size] = audio_record.audio_path.with_name(
        f"{audio_record.audio_path.stem}_{model_size}.json",
    )
    if audio_record.transcripts[model_size].exists():
        with audio_record.transcripts[model_size].open("r") as f:
            result = json.load(f)
        return models.Result(**result)

    logger.debug("loading model ...")
    model = whisper.load_model(model_size)

    logger.debug("transcribe ...")
    st = time.time()
    result = model.transcribe(str(audio_record.audio_path))
    el = time.time() - st
    logger.debug("elapsed time: %s", str(datetime.timedelta(seconds=el)))

    # store transcript
    with audio_record.transcripts[model_size].open("w") as f:
        json.dump(result, f)

    return models.Result(**result)


def get_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Get the overlap between two segments."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def convert_wav(audio_record: models.Record) -> None:
    """Convert an audio file to wav."""
    if audio_record.audio_path.suffix == ".wav":
        logger.debug("audio is in wav format!")
    wav_path = audio_record.audio_path.with_name(f"{audio_record.audio_path.stem}.wav")
    if wav_path.exists():
        logger.debug("%s exists!", audio_record.audio_path)
        audio_record.audio_path = wav_path

    logger.debug("loading %s", audio_record)
    sound = AudioSegment.from_file(audio_record.audio_path)

    logger.debug("exporting to % s", wav_path)
    sound.export(wav_path, format="wav")
    audio_record.audio_path = wav_path


def get_diarization(audio_record: models.Record, use_auth_token: str) -> Annotation:
    """Get the diarization from the audio file."""
    audio_record.diar_pkl = audio_record.audio_path.with_name(
        f"{audio_record.audio_path.stem}_diar.pkl",
    )
    if audio_record.diar_pkl.exists():
        logger.debug("loading %s", audio_record.diar_pkl)
        with audio_record.diar_pkl.open("rb") as handle:
            diarization = pickle.load(handle)  # noqa: S301
    else:
        convert_wav(audio_record)
        audio_record.diar_pkl = audio_record.audio_path.with_name(
            f"{audio_record.audio_path.stem}_diar.pkl",
        )
        logger.debug("loading model ...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=use_auth_token,
        )
        logger.debug("diarization ...")
        diarization = pipeline(audio_record.audio_path)

        # save diarization
        with audio_record.diar_pkl.open("wb") as handle:
            pickle.dump(diarization, handle, protocol=pickle.HIGHEST_PROTOCOL)

    audio_record.diar_json = audio_record.audio_path.with_name(
        f"{audio_record.audio_path.stem}_diar.json",
    )
    speaker_dict: dict[str, str] = {}
    segments_dict = get_segments(diarization, speaker_dict)
    with audio_record.diar_json.open("w") as f:
        json.dump(segments_dict, f)
        logger.debug("dumped diarization to %s", audio_record.diar_json)

    return diarization


def get_spoken_time(
    result: models.Result,
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
    result: models.Result,
    speakers_dict: dict[str, str],
    figs_path: str = "./data/logs",
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

        fig_path = f"{figs_path}/{speakers_dict[sp]}.png"
        figs.append(fig_path)
        plt.savefig(fig_path)
    return figs


def get_audio_format(audio_record: models.Record) -> str:
    """Given a path to an audio file, return the file format."""
    with audio_record.audio_path.open("rb") as f:
        audio_data = f.read()
    audio_format = magic.from_buffer(audio_data, mime=True)
    logger.debug("The audio file is in the format %s.", audio_format)
    return audio_format
