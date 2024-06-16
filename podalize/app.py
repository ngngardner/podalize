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


def audio_fingerprint_dir(audio_path: Path) -> Path:
    """Create the fingerprint dir for a given audio file."""
    fingerprint = utils.hash_audio_file(audio_path)
    dest = configs.podalize_path / fingerprint
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(audio_path, dest / audio_path.name)
    return dest / audio_path.name


def get_file_audio(uploaded_file: UploadedFile) -> Path:
    """Download an uploaded file to a destination folder."""
    audio_path = configs.tmp_path / str(uploaded_file.name)
    if not audio_path.exists():
        with audio_path.open("wb") as f:
            f.write(uploaded_file.getvalue())
    return audio_fingerprint_dir(audio_path)


def get_youtube_audio(youtube_url: str) -> Path:
    """Download a youtube video to a destination folder."""
    if "youtube.com" in youtube_url:
        audio_path = utils.youtube_downloader(youtube_url, configs.tmp_path)
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
        audio_path = utils.convert_wav(tmp_path)
        tmp_path.unlink()
    return audio_fingerprint_dir(audio_path)


def process_audio(audio_path: Path) -> tuple[Annotation, list[str], torch.Tensor, int]:
    """Process the audio file and create diarization and labels."""
    diarization = utils.get_diarization(audio_path, configs.use_auth_token)
    audio_path = utils.convert_wav(audio_path)
    labels = diarization.labels()
    logger.debug("speakers: %s", labels)
    y, sr = torchaudio.load(audio_path)
    logger.debug("audio shape: %s, sample rate: %s", y.shape, sr)
    return diarization, labels, y, sr


def handle_speakers(
    audio_path: Path,
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
        speaker_sample_path = audio_path.parent / f"{sp}.wav"
        waveform = y[:, s1:e1]
        torchaudio.save(speaker_sample_path, waveform, sr)
        st.audio(str(speaker_sample_path), format="audio/wav", start_time=0)
    return speakers_dict


def handle_segments(audio_path: Path) -> dict[tuple[float, float], str]:
    """Create the segments dictionary."""
    segments_path = audio_path.with_name(f"{audio_path.stem}_diar.json")
    try:
        with segments_path.open("rb") as f:
            segments = json.load(f)
    except UnicodeDecodeError:
        logger.debug(segments_path)
        logger.exception("Failed to decode diarization json file.")
        raise

    segments_dict = {}
    for key, v in segments.items():
        seg_key = [float(i) for i in key.split(",")]
        segments_dict[(seg_key[0], seg_key[1])] = v
    return segments_dict


def generate_figs(
    audio_path: Path,
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
    fig1.savefig(audio_path.parent / "spoken_time.png")
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
            audio_path = get_file_audio(uploaded_file)
        if youtube_url:
            audio_path = get_youtube_audio(youtube_url)

        diarization, labels, y, sr = process_audio(audio_path)
        with st.sidebar:
            speakers_dict = handle_speakers(audio_path, diarization, labels, y, sr)
        model_sizes = ["tiny", "small", "base", "medium", "large"]
        model_size = st.selectbox("Select Model Size", model_sizes, index=3)
        result = utils.get_transcript(
            model_size=model_size or "base",
            audio_path=audio_path,
        )

        segements_dict = handle_segments(audio_path)
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

        generate_figs(audio_path, speakers_dict, spoken_time_secs)

        pod_name = st.text_input("Enter Podcast Name", value=audio_path.name)
        st.download_button("Download transcript", transcript[3:])
        if st.button("Download"):
            handle_document(transcript, pod_name, speakers_dict)
