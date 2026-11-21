from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_huggingface_cache_is_set_before_library_import():
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("HF_HOME", None)
    check_cache = (
        "import app; "
        "from pathlib import Path; "
        "from app.utils.config import settings; "
        "from huggingface_hub import constants; "
        "assert Path(constants.HF_HOME).resolve() == Path(settings.hf_home).resolve()"
    )

    subprocess.run(
        [sys.executable, "-c", check_cache],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
