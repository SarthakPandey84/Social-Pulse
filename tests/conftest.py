"""Pytest session fixture ensuring clean-checkout builds have required local artifacts before tests run."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.build_all_artifacts import (
    artifacts_present_and_nonempty,
    build_all_artifacts,
    get_build_lock,
)


@pytest.fixture(scope="session", autouse=True)
def ensure_clean_checkout_artifacts_built() -> None:
    """Build deterministic derived artifacts on fresh clean checkouts where gitignored artifacts are absent."""
    sample_posts = ROOT_DIR / "data" / "sample" / "sample_posts.csv"
    if not sample_posts.is_file():
        return
    if artifacts_present_and_nonempty(ROOT_DIR):
        return
    with get_build_lock(ROOT_DIR):
        if not artifacts_present_and_nonempty(ROOT_DIR):
            build_all_artifacts(root_dir=ROOT_DIR)

