from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRATCH_ROOT = PROJECT_ROOT / ".tmp_test_runs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_scratch_root() -> Iterator[None]:
    shutil.rmtree(TEST_SCRATCH_ROOT, ignore_errors=True)
    yield
    shutil.rmtree(TEST_SCRATCH_ROOT, ignore_errors=True)


def _workspace_temp_dir(name: str) -> Iterator[Path]:
    TEST_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = "".join(part[:1] for part in name.split("-"))[:3] or "tmp"
    temp_dir = TEST_SCRATCH_ROOT / f"{prefix}-{uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_data_root() -> Iterator[Path]:
    for temp_dir in _workspace_temp_dir("data-root"):
        root = temp_dir / "data_root"
        root.mkdir(parents=True, exist_ok=True)
        yield root


@pytest.fixture
def temp_analysis_root() -> Iterator[Path]:
    for temp_dir in _workspace_temp_dir("analysis-root"):
        root = temp_dir / "analysis_outputs"
        root.mkdir(parents=True, exist_ok=True)
        yield root


@pytest.fixture
def temp_workspace_root() -> Iterator[Path]:
    for temp_dir in _workspace_temp_dir("workspace-root"):
        root = temp_dir / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        yield root
