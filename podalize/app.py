"""Main Podalize application module."""

import datetime
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torchaudio
from pyannote.core.annotation import Annotation
from streamlit.runtime.uploaded_file_manager import UploadedFile

from podalize import configs, utils
from podalize.document_generator import DocumentGenerator
from podalize.logger import get_logger

logger = get_logger(__name__)


def get_file_audio(uploaded_file: UploadedFile) -> Path:
    """Download an uploaded file to a destination folder."""
    p2audio = configs.audio_path / str(uploaded_file.name)
    if not p2audio.exists():
        with p2audio.open("wb") as f:
            f.write(uploaded_file.getvalue())
    return p2audio


def get_youtube_audio(youtube_url: str) -> Path:
    """Download a youtube video to a destination folder."""
    if "youtube.com" in youtube_url:
        p2audio = utils.youtube_downloader(youtube_url, configs.audio_path)
    else:
        path2out = configs.audio_path / "audio.unknown"
        youtube_dl_path = shutil.which("youtube-dl")
        if not youtube_dl_path:
            msg = "youtube-dl not found in PATH"
            raise RuntimeError(msg)
        subprocess.run(
            [youtube_dl_path, youtube_url, f"-o{path2out}"],  # noqa: S603
            check=False,
        )
        p2audio = utils.audio2wav(path2out)
        path2out.unlink()
    return p2audio


def process_audio(p2audio: Path) -> tuple[Annotation, list[str], torch.Tensor, int]:
    """Process the audio file and create diarization and labels."""
    diarization = utils.get_diarization(p2audio, configs.use_auth_token)
    p2audio = utils.audio2wav(p2audio)
    labels = diarization.labels()
    logger.debug("speakers: %s", labels)
    y, sr = torchaudio.load(p2audio)
    logger.debug("audio shape: %s, sample rate: %s", y.shape, sr)
    return diarization, labels, y, sr


def handle_speakers(
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
        path2sp = f"{configs.audio_path}/{sp}.wav"
        waveform = y[:, s1:e1]
        torchaudio.save(path2sp, waveform, sr)
        st.audio(path2sp, format="audio/wav", start_time=0)
    return speakers_dict


def handle_segments(p2audio: Path) -> dict[tuple[float, float], str]:
    """Create the segments dictionary."""
    p2s = p2audio.with_name(f"{p2audio.stem}_diar.json")
    try:
        with p2s.open("rb") as f:
            segments = json.load(f)
    except UnicodeDecodeError:
        logger.debug(p2s)
        logger.exception("Failed to decode diarization json file.")
        raise

    segments_dict = {}
    for key, v in segments.items():
        seg_key = [float(i) for i in key.split(",")]
        segments_dict[(seg_key[0], seg_key[1])] = v
    return segments_dict


def generate_figs(
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
    fig1.savefig(f"{configs.log_path}/spoken_time.png")
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
        path2logs=configs.log_path,
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
    path2pdf = f"{configs.log_path}/podalize_{pod_name}"
    rg.doc.generate_pdf(path2pdf, clean_tex=False)
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
            p2audio = get_file_audio(uploaded_file)
        if youtube_url:
            p2audio = get_youtube_audio(youtube_url)

        diarization, labels, y, sr = process_audio(p2audio)
        with st.sidebar:
            speakers_dict = handle_speakers(diarization, labels, y, sr)
        model_sizes = ["tiny", "small", "base", "medium", "large"]
        model_size = st.selectbox("Select Model Size", model_sizes, index=3)
        result = utils.get_transcript(
            model_size=model_size or "base",
            path2audio=p2audio,
        )

        segements_dict = handle_segments(p2audio)
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

        generate_figs(speakers_dict, spoken_time_secs)

        pod_name = st.text_input("Enter Podcast Name", value=p2audio.name)
        st.download_button("Download transcript", transcript[3:])
        if st.button("Download"):
            handle_document(transcript, pod_name, speakers_dict)
