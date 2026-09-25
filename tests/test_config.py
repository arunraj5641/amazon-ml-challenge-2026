"""Tests for configuration loading, path resolution, and logger setup."""

import logging
import os
from pathlib import Path
import pytest

from src.config import get_project_root, load_config, ProjectConfig
from src.utils.logger import setup_logger


def test_get_project_root():
    root = get_project_root()
    assert root.exists()
    assert (root / "src").is_dir()
    assert (root / "configs").is_dir()


def test_load_default_config():
    config = load_config()
    assert isinstance(config, ProjectConfig)
    assert config.project_name == "amazon-ml-challenge-2026"
    assert config.phase == 0
    assert config.environment_mode in {"local", "sage"}
    assert config.paths.root.exists()
    assert config.paths.data_dir.exists()
    assert config.paths.artifacts_dir.exists()
    assert config.paths.configs_dir.exists()
    assert config.paths.experiments_dir.exists()
    assert config.paths.results_csv.exists()


def test_validate_paths():
    config = load_config()
    status = config.validate_paths(create_missing=False)
    assert status["root"] is True
    assert status["data_dir"] is True
    assert status["artifacts_dir"] is True
    assert status["configs_dir"] is True
    assert status["experiments_dir"] is True
    assert status["results_csv"] is True


def test_environment_variable_overrides(monkeypatch):
    monkeypatch.setenv("PROJECT_NAME", "custom-amazon-ml")
    monkeypatch.setenv("PHASE", "1")
    monkeypatch.setenv("ENVIRONMENT", "sage")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    config = load_config()
    assert config.project_name == "custom-amazon-ml"
    assert config.phase == 1
    assert config.environment_mode == "sage"
    assert config.log_level == "DEBUG"


def test_logger_setup():
    logger = setup_logger(name="test_logger", level="WARNING")
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1

    # Calling again should not add duplicate handlers
    logger_duplicate = setup_logger(name="test_logger", level="WARNING")
    assert len(logger_duplicate.handlers) == 1
