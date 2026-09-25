"""Unit tests for validate_data CLI script."""

import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_data"
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_data.py"


def test_cli_valid_sample(tmp_path: Path):
    """Verify CLI returns exit code 0 on valid sample data."""
    out_dir = tmp_path / "val_output"
    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--input",
        str(SAMPLE_DATA_DIR),
        "--output",
        str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Validation PASSED successfully!" in res.stdout
    assert (out_dir / "validation_report.json").is_file()
    assert (out_dir / "dataset_stats.json").is_file()


def test_cli_invalid_dataset_fails(tmp_path: Path):
    """Verify CLI returns non-zero exit code when required files are missing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--input",
        str(empty_dir),
        "--output",
        str(tmp_path / "out"),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Validation failed" in res.stdout or "Validation failed" in res.stderr
