"""Podalize configuration module."""

from pathlib import Path

path2audios = str(Path("./data"))
path2logs = str(Path("./data/logs"))
Path(path2audios).mkdir(exist_ok=True)
Path(path2logs).mkdir(exist_ok=True)

# printing verbose
verbose = False

# pyannote.audio api access token
#
# visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and
# accept user conditions. visit hf.co/settings/tokens to create an access token.
# set use_auth_token using the token here or store it in a file named api.token
# to be loaded.
with Path("api.token").open() as f:
    use_auth_token = f.read()
use_auth_token = use_auth_token.rstrip("\n")
