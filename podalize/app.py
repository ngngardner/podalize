"""Main Podalize application module."""

import datetime
import json
import shutil
import subprocess

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torchaudio
from pyannote.core.annotation import Annotation
from streamlit.runtime.uploaded_file_manager import UploadedFile

from podalize import configs, db, models, utils
from podalize.document_generator import DocumentGenerator
from podalize.logger import get_logger

logger = get_logger(__name__)


def audio_fingerprint_dir(audio_record: models.Record) -> None:
    """Create the fingerprint dir for a given audio file."""
    fingerprint = utils.hash_audio_file(audio_record)
    dest = configs.podalize_path / fingerprint
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(audio_record.audio_path, dest / audio_record.audio_path.name)

    audio_record.file_dir = dest
    audio_record.audio_path = dest / audio_record.audio_path.name


def get_file_audio(uploaded_file: UploadedFile) -> models.Record:
    """Download an uploaded file to a destination folder."""
    audio_record = models.Record(
        audio_path=configs.tmp_path / str(uploaded_file.name),
        file_dir=configs.tmp_path,
    )
    if not audio_record.audio_path.exists():
        with audio_record.audio_path.open("wb") as f:
            f.write(uploaded_file.getvalue())
    audio_fingerprint_dir(audio_record)
    return audio_record


def get_youtube_audio(youtube_url: str) -> models.YoutubeRecord:
    """Download a youtube video to a destination folder."""
    if "youtube.com" in youtube_url:
        audio_record = utils.youtube_downloader(youtube_url)
    else:
        tmp_path = configs.tmp_path / "audio.unknown"
        youtube_dl_path = shutil.which("youtube-dl")
        if not youtube_dl_path:
            msg = "youtube-dl not found in PATH"
            raise RuntimeError(msg)
        subprocess.run(
            [youtube_dl_path, youtube_url, f"-o{tmp_path}"],  # noqa: S603
            check=False,
        )
        audio_record = models.YoutubeRecord(
            video_url=youtube_url,
            audio_path=tmp_path,
            file_dir=configs.tmp_path,
        )
        utils.convert_wav(audio_record)
        tmp_path.unlink()
    audio_fingerprint_dir(audio_record)
    db.store_youtube_record(audio_record)
    return audio_record


def process_audio(
    audio_record: models.Record,
) -> tuple[Annotation, list[str], torch.Tensor, int]:
    """Process the audio file and create diarization and labels."""
    diarization = utils.get_diarization(audio_record, configs.use_auth_token)
    utils.convert_wav(audio_record)
    labels = diarization.labels()
    logger.debug("speakers: %s", labels)
    y, sr = torchaudio.load(audio_record.audio_path)
    logger.debug("audio shape: %s, sample rate: %s", y.shape, sr)
    return diarization, labels, y, sr


def handle_speakers(
    audio_record: models.Record,
    diarization: Annotation,
    labels: list[str],
    y: torch.Tensor,
    sr: int,
) -> dict[str, str]:
    """Create the speakers dictionary based on the diarization."""
    speakers_dict = {}
    for ii, sp in enumerate(labels):
        speakers_dict[sp] = st.text_input(f"Speaker_{ii}", sp)
        s, e, _ = utils.get_largest_duration(diarization, sp)
        s1 = int(s * sr)
        e1 = int(e * sr)
        speaker_sample_path = audio_record.audio_path.parent / f"{sp}.wav"
        waveform = y[:, s1:e1]
        torchaudio.save(speaker_sample_path, waveform, sr)
        st.audio(str(speaker_sample_path), format="audio/wav", start_time=0)
    return speakers_dict


def handle_segments(audio_record: models.Record) -> dict[tuple[float, float], str]:
    """Create the segments dictionary."""
    if not audio_record.diar_json:
        audio_record.diar_json = audio_record.audio_path.with_name(
            f"{audio_record.audio_path.stem}_diar.json",
        )
    try:
        with audio_record.diar_json.open("rb") as f:
            segments = json.load(f)
    except UnicodeDecodeError:
        logger.debug(audio_record.diar_json)
        logger.exception("Failed to decode diarization json file.")
        raise

    segments_dict = {}
    for key, v in segments.items():
        seg_key = [float(i) for i in key.split(",")]
        segments_dict[(seg_key[0], seg_key[1])] = v
    return segments_dict


def generate_figs(
    audio_record: models.Record,
    speakers_dict: dict[str, str],
    spoken_time_secs: dict[str, float],
) -> None:
    """Generate figures for spoken time."""
    st.header("Analyze")
    st.subheader("Spoken Time")
    labels = list(speakers_dict.values())
    sizes = spoken_time_secs.values()
    sizes_str = [str(datetime.timedelta(seconds=round(s, 0))) for s in sizes]
    labels = [f"{label},\n{z}" for label, z in zip(labels, sizes_str, strict=False)]
    explode = (0.05,) * len(labels)
    fig1, ax1 = plt.subplots()
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax1.axis("equal")
    fig1.savefig(audio_record.audio_path.parent / "spoken_time.png")
    st.pyplot(fig1)


def handle_document(
    transcript: str,
    pod_name: str,
    speakers_dict: dict[str, str],
) -> None:
    """Create a .pdf document for download based on the generated transcript."""
    spoken_fig = configs.log_path.glob("spoken*.png")
    all_figs = configs.log_path.glob("$*.png")
    wc_figs = [
        f for f in all_figs if [v for v in speakers_dict.values() if v in str(f)]
    ]

    rg = DocumentGenerator(
        title=pod_name,
        author="Created by Podalize",
        log_path=configs.log_path,
    )

    for f in spoken_fig:
        rg.add_image(str(f), caption="Percentage of spoken time per speaker")
        rg.add_new_page()

    for f in wc_figs:
        rg.add_image(str(f), caption="Word cloud per speaker")
        rg.add_new_page()

    output = transcript[3:]
    rg.add_section("Transcript", output)
    logger.debug("number of figures: %s", rg.fig_count)
    pdf_path = f"{configs.log_path}/podalize_{pod_name}"
    rg.doc.generate_pdf(pdf_path, clean_tex=False)
    logger.debug("podalized!")


def app() -> None:
    """Run main application."""
    if torch.cuda.is_available():
        st.text(f"cuda device: {torch.cuda.get_device_name()}")

    st.title("Podalize: podcast transcription and analysis")
    uploaded_file = st.file_uploader("Choose an audio", type=["mp3", "wav"])
    youtube_url = st.text_input("Youtube/Podcast URL")
    if uploaded_file or youtube_url:
        st.spinner(text="In progress...")
        if uploaded_file:
            audio_record = get_file_audio(uploaded_file)
        if youtube_url:
            audio_record = get_youtube_audio(youtube_url)

        diarization, labels, y, sr = process_audio(audio_record)
        with st.sidebar:
            speakers_dict = handle_speakers(audio_record, diarization, labels, y, sr)
        model_sizes = ["tiny", "small", "base", "medium", "large"]
        model_size = st.selectbox("Select Model Size", model_sizes, index=3)
        result = utils.get_transcript(
            model_size=model_size or "base",
            audio_record=audio_record,
        )

        segements_dict = handle_segments(audio_record)
        transcript = utils.merge_tran_diar(result, segements_dict, speakers_dict)
        st.subheader("Transcript")
        st.text_area(
            label="transcript",
            value=transcript,
            label_visibility="hidden",
            height=512,
        )

        speakers = list(speakers_dict.keys())
        spoken_time, spoken_time_secs = utils.get_spoken_time(result, speakers)

        generate_figs(audio_record, speakers_dict, spoken_time_secs)

        pod_name = st.text_input(
            "Enter Podcast Name",
            value=audio_record.audio_path.name,
        )
        st.download_button("Download transcript", transcript[3:])
        if st.button("Download"):
            handle_document(transcript, pod_name, speakers_dict)
