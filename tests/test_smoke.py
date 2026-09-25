"""Tests verifying the smoke test script and Phase 0 setup criteria."""

import csv
from pathlib import Path
from scripts.smoke_test import run_smoke_test, EXPECTED_RESULTS_HEADER
from src.config import get_project_root


def test_smoke_test_execution():
    """Verify that smoke test passes cleanly."""
    success = run_smoke_test()
    assert success is True


def test_results_csv_header():
    """Ensure experiments/results.csv matches the required format."""
    root = get_project_root()
    results_path = root / "experiments" / "results.csv"
    assert results_path.is_file()

    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == EXPECTED_RESULTS_HEADER


def test_gitignore_contains_required_rules():
    """Ensure .gitignore excludes virtual envs, secrets, cache, and raw artifacts."""
    root = get_project_root()
    gitignore_path = root / ".gitignore"
    assert gitignore_path.is_file()

    content = gitignore_path.read_text(encoding="utf-8")
    assert ".venv/" in content
    assert "__pycache__/" in content
    assert "*.pyc" in content or "*.py[cod]" in content
    assert ".env" in content
    assert ".DS_Store" in content
    assert ".ipynb_checkpoints/" in content
    assert "data/*" in content
    assert "artifacts/*" in content
