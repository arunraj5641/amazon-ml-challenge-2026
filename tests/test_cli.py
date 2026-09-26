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


NORMALIZE_SCRIPT = REPO_ROOT / "scripts" / "normalize_data.py"


def test_normalize_cli_valid_sample(tmp_path: Path):
    """Verify normalize CLI runs cleanly and produces output TSVs and report."""
    out_dir = tmp_path / "norm_output"
    cmd = [
        sys.executable,
        str(NORMALIZE_SCRIPT),
        "--input",
        str(SAMPLE_DATA_DIR),
        "--output",
        str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Normalization PASSED successfully!" in res.stdout
    assert (out_dir / "normalization_report.json").is_file()
    assert (out_dir / "train" / "train_source1_normalized.tsv").is_file()
    assert (out_dir / "test" / "test_source3_normalized.tsv").is_file()


def test_normalize_cli_invalid_dataset_fails(tmp_path: Path):
    """Verify normalize CLI exits with non-zero when required source files are missing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    cmd = [
        sys.executable,
        str(NORMALIZE_SCRIPT),
        "--input",
        str(empty_dir),
        "--output",
        str(tmp_path / "out"),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Normalization failed with errors" in res.stdout or "Normalization failed with errors" in res.stderr


EVALUATE_BLOCKING_SCRIPT = REPO_ROOT / "scripts" / "evaluate_blocking.py"
GENERATE_CANDIDATES_SCRIPT = REPO_ROOT / "scripts" / "generate_candidates.py"


def test_evaluate_blocking_cli_valid_sample(tmp_path: Path):
    """Verify evaluate_blocking CLI runs cleanly and produces artifacts."""
    out_dir = tmp_path / "block_eval_out"
    cmd = [
        sys.executable,
        str(EVALUATE_BLOCKING_SCRIPT),
        "--raw-input",
        str(SAMPLE_DATA_DIR),
        "--norm-input",
        str(SAMPLE_DATA_DIR),
        "--output",
        str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Phase 4 Blocking Strategy Evaluation COMPLETE!" in res.stdout
    assert (out_dir / "candidate_pairs.tsv").is_file()
    assert (out_dir / "blocking_stats.json").is_file()
    assert (out_dir / "blocking_strategy_results.json").is_file()
    assert (out_dir / "blocking_validation_report.json").is_file()
    assert (out_dir / "blocking_metadata.json").is_file()


def test_generate_candidates_cli_valid_sample(tmp_path: Path):
    """Verify generate_candidates CLI runs cleanly and produces candidate_pairs.tsv."""
    out_dir = tmp_path / "gen_cands_out"
    cmd = [
        sys.executable,
        str(GENERATE_CANDIDATES_SCRIPT),
        "--norm-input",
        str(SAMPLE_DATA_DIR),
        "--output",
        str(out_dir),
        "--strategy",
        "composite",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Phase 4 Candidate Generation COMPLETE!" in res.stdout
    assert (out_dir / "candidate_pairs.tsv").is_file()
    assert (out_dir / "blocking_stats.json").is_file()

