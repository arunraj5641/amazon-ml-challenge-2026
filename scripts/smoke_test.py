#!/usr/bin/env python3
"""Smoke test script for Amazon ML Challenge 2026 Phase 0.

Verifies:
1. Expected project directory structure exists.
2. Required setup and documentation files exist.
3. experiments/results.csv has the required schema.
4. Project configuration loads correctly and validates paths.
5. Logging infrastructure initializes cleanly.
"""

import csv
import sys
from pathlib import Path

# Add project root to sys.path if not already present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_project_root, load_config
from src.utils.logger import setup_logger

EXPECTED_DIRECTORIES = [
    "src",
    "scripts",
    "tests",
    "configs",
    "docs",
    "data",
    "artifacts",
    "experiments",
]

EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "data/README.md",
    "docs/README.md",
    "experiments/results.csv",
    "configs/default.yaml",
]

EXPECTED_RESULTS_HEADER = [
    "experiment_id",
    "git_commit",
    "blocking_version",
    "feature_version",
    "model_version",
    "threshold",
    "precision",
    "recall",
    "f0_5",
    "leaderboard_score",
    "notes",
]


def run_smoke_test() -> bool:
    """Runs all verification checks.

    Returns:
        True if all checks pass, False otherwise.
    """
    logger = setup_logger(name="smoke_test", level="INFO")
    logger.info("Starting Phase 0 Smoke Test...")

    root = get_project_root()
    logger.info("Project Root detected: %s", root)
    failures = []

    # 1. Verify directories
    for d in EXPECTED_DIRECTORIES:
        p = root / d
        if p.is_dir():
            logger.info("  [PASS] Directory exists: %s", d)
        else:
            logger.error("  [FAIL] Missing directory: %s", d)
            failures.append(f"Directory missing: {d}")

    # 2. Verify files
    for f in EXPECTED_FILES:
        p = root / f
        if p.is_file():
            logger.info("  [PASS] File exists: %s", f)
        else:
            logger.error("  [FAIL] Missing file: %s", f)
            failures.append(f"File missing: {f}")

    # 3. Verify experiments/results.csv header
    results_path = root / "experiments" / "results.csv"
    if results_path.is_file():
        try:
            with open(results_path, "r", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                header = next(reader, None)
                if header == EXPECTED_RESULTS_HEADER:
                    logger.info("  [PASS] experiments/results.csv schema matches expected header.")
                else:
                    logger.error("  [FAIL] experiments/results.csv header mismatch. Found: %s", header)
                    failures.append("experiments/results.csv header mismatch")
        except Exception as e:
            logger.error("  [FAIL] Could not read experiments/results.csv: %s", e)
            failures.append(f"results.csv read error: {e}")

    # 4. Verify configuration loading
    try:
        config = load_config()
        logger.info("  [PASS] Configuration loaded successfully.")
        logger.info("         Project Name: %s", config.project_name)
        logger.info("         Phase: %d", config.phase)
        logger.info("         Environment: %s", config.environment_mode)
        logger.info("         Log Level: %s", config.log_level)

        path_status = config.validate_paths()
        for path_name, exists in path_status.items():
            if exists:
                logger.info("  [PASS] Configured path verified: %s", path_name)
            else:
                logger.error("  [FAIL] Configured path missing: %s", path_name)
                failures.append(f"Configured path missing: {path_name}")

    except Exception as e:
        logger.error("  [FAIL] Failed to load configuration: %s", e)
        failures.append(f"Config load error: {e}")

    if failures:
        logger.error("Smoke test FAILED with %d error(s):", len(failures))
        for err in failures:
            logger.error("  - %s", err)
        return False

    logger.info("Smoke test PASSED! Project infrastructure is ready for Phase 0.")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
