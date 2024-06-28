"""Podalize configuration module."""

import json
import os
from pathlib import Path

from podalize.logger import get_logger

logger = get_logger(__name__)

temp: Path | str | None = os.environ.get("PODALIZE_PATH", None)
if temp is None or not Path(temp).exists():
    podalize_path = Path("~/.podalize").expanduser()
    logger.warning(
        "PODALIZE_PATH variable not found or corrupt. Creating %s",
        podalize_path,
    )
    podalize_path.mkdir(exist_ok=True)
else:
    podalize_path = Path(temp)
log_path = podalize_path / "logs"
tmp_path = podalize_path / "tmp"
log_path.mkdir(exist_ok=True)
tmp_path.mkdir(exist_ok=True)

# TODO: implement sqlite database for storing fingerprint relations and
# eventually generated data
db_path = podalize_path / "db.json"
if not db_path.exists():
    with db_path.open("w") as f:
        json.dump({}, f)
with db_path.open() as f:
    db: dict[str, str] = json.load(f)

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
