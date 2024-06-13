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
from podalize.DocumentGenerator import DocumentGenerator
from podalize.logger import get_logger

logger = get_logger(__name__)


def get_file_audio(uploaded_file: UploadedFile) -> str:
    """Download an uploaded file to a destination folder."""
    p2audio = Path(configs.path2audios) / uploaded_file.name
    if not Path(p2audio).exists():
        with Path(p2audio).open("wb") as f:
            f.write(uploaded_file.getvalue())
    return str(p2audio)


def get_youtube_audio(youtube_url: str) -> str:
    """Download a youtube video to a destination folder."""
    if "youtube.com" in youtube_url:
        p2audio = utils.youtube_downloader(youtube_url, configs.path2audios)
    else:
        path2out = str(Path(configs.path2audios / "audio.unknown"))
        youtube_dl_path = shutil.which("youtube-dl")
        subprocess.run(
            [youtube_dl_path, youtube_url, f"-o{path2out}"],  # noqa: S603
            check=False,
        )
        p2audio = utils.audio2wav(path2out)
        Path(path2out).unlink()
    return str(p2audio)


def process_audio(p2audio: str) -> (Annotation, dict, torch.Tensor, int):
    """Process the audio file and create diarization and labels."""
    p2audio = str(p2audio)
    diarization = utils.get_diarization(p2audio, configs.use_auth_token)
    p2audio = utils.audio2wav(p2audio)
    labels = diarization.labels()
    logger.debug("speakers: %s", labels)
    y, sr = torchaudio.load(p2audio)
    logger.debug("audio shape: %s, sample rate: %s", y.shape, sr)
    return diarization, labels, y, sr


def handle_speakers(
    diarization: Annotation,
    labels: dict,
    y: torch.Tensor,
    sr: int,
) -> dict:
    """Create the speakers dictionary based on the diarization."""
    speakers_dict = {}
    for ii, sp in enumerate(labels):
        speakers_dict[sp] = st.text_input(f"Speaker_{ii}", sp)
        s, e, _ = utils.get_larget_duration(diarization, sp)
        s1 = int(s * sr)
        e1 = int(e * sr)
        path2sp = f"{configs.path2audios}/{sp}.wav"
        waveform = y[:, s1:e1]
        torchaudio.save(path2sp, waveform, sr)
        st.audio(path2sp, format="audio/wav", start_time=0)
    return speakers_dict


def handle_segments(p2audio: str) -> dict:
    """Create the segments dictionary."""
    p2s = p2audio.replace(".wav", "_diar.json")
    with Path(p2s).open() as f:
        segments = json.load(f)

    segments_dict = {}
    for key, v in segments.items():
        seg_key = [float(i) for i in key.split(",")]
        segments_dict[(seg_key[0], seg_key[1])] = v
    return segments_dict


def generate_figs(speakers_dict: dict, spoken_time_secs: dict) -> None:
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
    fig1.savefig(f"{configs.path2logs}/spoken_time.png")
    st.pyplot(fig1)


def handle_document(transcript: str, pod_name: str, speakers_dict: dict) -> None:
    """Create a .pdf document for download based on the generated transcript."""
    spoken_fig = Path(configs.path2logs).glob("spoken*.png")
    all_figs = Path(configs.path2logs).glob("$*.png")
    wc_figs = [f for f in all_figs if [v for v in speakers_dict.values() if v in f]]

    args = {
        "title": pod_name,
        "author": "Created by Podalize",
        "path2logs": configs.path2logs,
    }
    rg = DocumentGenerator(**args)

    for f in spoken_fig:
        rg.add_image(f, caption="Percentage of spoken time per speaker")
        rg.add_new_page()

    for f in wc_figs:
        rg.add_image(f, caption="Word cloud per speaker")
        rg.add_new_page()

    output = transcript[3:]
    rg.add_section("Transcript", output)
    logger.debug("number of figures: %s", rg.fig_count)
    path2pdf = f"{configs.path2logs}/podalize_{pod_name}"
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
        result = utils.get_transcript(model_size=model_size, path2audio=p2audio)

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

        pod_name = st.text_input("Enter Podcast Name", value=Path(p2audio).name)
        st.download_button("Download transcript", transcript[3:])
        if st.button("Download"):
            handle_document(pod_name, speakers_dict)
