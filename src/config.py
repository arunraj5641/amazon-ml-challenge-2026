"""Configuration loader and environment management.

Supports loading YAML configs, loading .env files, resolving paths relative
to the project root, and overriding settings via environment variables.
Works both locally and in Sage / SageMaker environments without hardcoded paths.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def get_project_root() -> Path:
    """Returns the absolute Path to the repository root."""
    override = os.getenv("PROJECT_ROOT")
    if override:
        return Path(override).resolve()

    # Default: 2 levels up from src/config.py
    current = Path(__file__).resolve().parent.parent
    return current


@dataclass
class ProjectPaths:
    """Project directory paths resolved to absolute paths."""
    root: Path
    data_dir: Path
    artifacts_dir: Path
    configs_dir: Path
    experiments_dir: Path
    results_csv: Path


@dataclass
class S3Config:
    """S3 storage configuration."""
    enabled: bool
    bucket_name: Optional[str]
    region: str
    data_prefix: str
    artifacts_prefix: str


@dataclass
class ProjectConfig:
    """Consolidated project configuration."""
    project_name: str
    phase: int
    environment_mode: str  # 'local' or 'sage'
    log_level: str
    paths: ProjectPaths
    storage: S3Config = field(default_factory=lambda: S3Config(
        enabled=False,
        bucket_name=None,
        region="ap-southeast-2",
        data_prefix="raw/dataset",
        artifacts_prefix="artifacts/validation",
    ))
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def validate_paths(self, create_missing: bool = False) -> Dict[str, bool]:
        """Checks if configured directories exist.

        Args:
            create_missing: If True, creates missing directories.

        Returns:
            Dict mapping path attribute names to boolean existence status.
        """
        status = {}
        dirs_to_check = {
            "root": self.paths.root,
            "data_dir": self.paths.data_dir,
            "artifacts_dir": self.paths.artifacts_dir,
            "configs_dir": self.paths.configs_dir,
            "experiments_dir": self.paths.experiments_dir,
        }
        for name, p in dirs_to_check.items():
            if create_missing and not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            status[name] = p.exists()

        status["results_csv"] = self.paths.results_csv.exists()
        return status


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """Loads YAML configuration with environment variable overrides.

    Args:
        config_path: Path to YAML config file. If None, defaults to
            configs/default.yaml under the project root.

    Returns:
        ProjectConfig object.
    """
    root = get_project_root()

    # Load .env if present
    env_file = root / ".env"
    if env_file.exists() and load_dotenv:
        load_dotenv(dotenv_path=env_file)

    # Determine config file path
    if config_path is None:
        env_config = os.getenv("CONFIG_PATH")
        if env_config:
            target_config = Path(env_config)
            if not target_config.is_absolute():
                target_config = root / target_config
        else:
            target_config = root / "configs" / "default.yaml"
    else:
        target_config = Path(config_path)
        if not target_config.is_absolute():
            target_config = root / target_config

    raw: Dict[str, Any] = {}
    if target_config.exists():
        with open(target_config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    project_section = raw.get("project", {})
    env_section = raw.get("environment", {})
    paths_section = raw.get("paths", {})

    # Extract settings with environment variable overrides
    project_name = os.getenv("PROJECT_NAME", project_section.get("name", "amazon-ml-challenge-2026"))
    phase = int(os.getenv("PHASE", project_section.get("phase", 0)))
    environment_mode = os.getenv("ENVIRONMENT", env_section.get("mode", "local")).lower()
    log_level = os.getenv("LOG_LEVEL", env_section.get("log_level", "INFO")).upper()

    def resolve_path(rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (root / p).resolve()

    data_dir = resolve_path(os.getenv("PROJECT_DATA_DIR", paths_section.get("data_dir", "data")))
    artifacts_dir = resolve_path(os.getenv("PROJECT_ARTIFACTS_DIR", paths_section.get("artifacts_dir", "artifacts")))
    configs_dir = resolve_path(os.getenv("PROJECT_CONFIGS_DIR", paths_section.get("configs_dir", "configs")))
    experiments_dir = resolve_path(os.getenv("PROJECT_EXPERIMENTS_DIR", paths_section.get("experiments_dir", "experiments")))
    results_csv = resolve_path(os.getenv("PROJECT_RESULTS_CSV", paths_section.get("results_csv", "experiments/results.csv")))

    resolved_paths = ProjectPaths(
        root=root,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        configs_dir=configs_dir,
        experiments_dir=experiments_dir,
        results_csv=results_csv,
    )

    storage_section = raw.get("storage", {})
    s3_section = storage_section.get("s3", {})
    s3_config = S3Config(
        enabled=os.getenv("S3_ENABLED", str(s3_section.get("enabled", False))).lower() in ("true", "1", "yes"),
        bucket_name=os.getenv("S3_BUCKET_NAME", s3_section.get("bucket_name")),
        region=os.getenv("AWS_REGION", s3_section.get("region", "ap-southeast-2")),
        data_prefix=os.getenv("S3_DATA_PREFIX", s3_section.get("data_prefix", "raw/dataset")),
        artifacts_prefix=os.getenv("S3_ARTIFACTS_PREFIX", s3_section.get("artifacts_prefix", "artifacts/validation")),
    )

    return ProjectConfig(
        project_name=project_name,
        phase=phase,
        environment_mode=environment_mode,
        log_level=log_level,
        paths=resolved_paths,
        storage=s3_config,
        raw_config=raw,
    )
