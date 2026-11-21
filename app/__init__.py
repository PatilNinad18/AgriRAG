"""AgriTrackr AI application package."""

import os

from app.utils.config import settings

# Hugging Face reads this setting when its libraries are imported.
os.environ.setdefault("HF_HOME", settings.hf_home)
