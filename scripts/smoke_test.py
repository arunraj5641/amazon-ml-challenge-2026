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

    # 5. Verify Phase 1 Data Ingestion & Validation pipeline on sample data
    sample_dir = root / "tests" / "fixtures" / "sample_data"
    if sample_dir.is_dir():
        try:
            from src.data.data_source import LocalDataSource
            from src.data.validator import DataValidator

            ds = LocalDataSource(base_dir=sample_dir)
            validator = DataValidator(data_source=ds, chunksize=100, logger=logger)
            val_result = validator.validate_all()
            if val_result.status == "PASS":
                logger.info("  [PASS] Phase 1 sample validation pipeline executed and PASSED.")
            else:
                logger.error("  [FAIL] Phase 1 sample validation returned FAIL status.")
                failures.append("Phase 1 sample validation failed")
        except Exception as e:
            logger.error("  [FAIL] Phase 1 sample validation error: %s", e)
            failures.append(f"Phase 1 sample validation error: {e}")

    # 6. Verify Phase 2 Data Normalization pipeline on sample data
    if sample_dir.is_dir():
        import tempfile
        try:
            from src.data.data_source import LocalDataSource
            from src.normalization.pipeline import NormalizationPipeline, FINAL_COLUMN_ORDER

            with tempfile.TemporaryDirectory() as tmp_out:
                in_ds = LocalDataSource(base_dir=sample_dir)
                out_ds = LocalDataSource(base_dir=tmp_out)
                norm_pipeline = NormalizationPipeline(
                    input_source=in_ds,
                    output_source=out_ds,
                    chunksize=100,
                    logger=logger,
                )
                norm_report = norm_pipeline.run()

                if norm_report["status"] == "PASS":
                    # Check that output file exists, columns match, and row count matches
                    norm_s1 = out_ds.read_table("train/train_source1_normalized.tsv")
                    raw_s1 = in_ds.read_table("train/train_source1.tsv")
                    assert len(norm_s1) == len(raw_s1)
                    assert list(norm_s1.columns) == FINAL_COLUMN_ORDER
                    assert list(norm_s1["entity_id"]) == list(raw_s1["entity_id"])
                    logger.info("  [PASS] Phase 2 sample normalization pipeline executed and PASSED.")
                else:
                    logger.error("  [FAIL] Phase 2 normalization returned FAIL status.")
                    failures.append("Phase 2 normalization failed")
        except Exception as e:
            logger.error("  [FAIL] Phase 2 normalization error: %s", e)
            failures.append(f"Phase 2 normalization error: {e}")

    # 7. Verify Phase 3 Training Pair Builder on sample data
    if sample_dir.is_dir():
        import tempfile
        try:
            from src.data.data_source import LocalDataSource
            from src.pairs.builder import TrainingPairBuilder, TrainingPairConfig
            from src.pairs.profiler import GroundTruthProfiler

            with tempfile.TemporaryDirectory() as tmp_pairs:
                in_ds = LocalDataSource(base_dir=sample_dir)
                out_ds = LocalDataSource(base_dir=tmp_pairs)

                # Profiler check
                profiler = GroundTruthProfiler(data_source=in_ds, logger=logger)
                profile = profiler.profile()
                assert profile.total_source1_entities == 4
                assert profile.total_positive_pairs == 3

                # Builder & Validator check
                builder = TrainingPairBuilder(
                    input_source=in_ds,
                    output_source=out_ds,
                    config=TrainingPairConfig(negatives_per_positive=1, seed=42),
                    logger=logger,
                )
                pairs_df, stats = builder.build_pairs_from_ground_truth(
                    candidate_pool_s2=["s2_101", "s2_102", "s2_105"],
                    candidate_pool_s3=["s3_101", "s3_106"],
                )
                assert stats["positive_pairs"] == 3
                assert stats["negative_pairs"] == 3

                val_res = builder.validator.validate(
                    pairs_df,
                    authoritative_positives={("s1_101", "s2_101"), ("s1_101", "s3_101"), ("s1_102", "s2_102")},
                )
                assert val_res.status == "PASS"

                saved = builder.save_artifacts(pairs_df, stats, validation_result=val_res)
                assert out_ds.exists("pair_stats.json")
                assert out_ds.exists("pair_generation_report.json")
                assert out_ds.exists("pair_validation_report.json")
                assert out_ds.exists("metadata.json")
                logger.info("  [PASS] Phase 3 sample pair builder pipeline executed and PASSED.")
        except Exception as e:
            logger.error("  [FAIL] Phase 3 pair builder error: %s", e)
            failures.append(f"Phase 3 pair builder error: {e}")

    # 8. Verify Phase 4 Candidate Blocking pipeline on sample data
    if sample_dir.is_dir():
        import tempfile
        try:
            from src.blocking.config import BlockingConfig
            from src.blocking.evaluator import BlockingEvaluator
            from src.blocking.io import save_all_blocking_artifacts
            from src.blocking.validator import BlockingValidator
            from src.data.data_source import LocalDataSource

            with tempfile.TemporaryDirectory() as tmp_blocking:
                in_ds = LocalDataSource(base_dir=sample_dir)
                out_ds = LocalDataSource(base_dir=tmp_blocking)

                b_cfg = BlockingConfig(
                    strategies=["exact_name", "name_token", "composite"],
                    max_posting_list_size=50,
                    seed=42,
                )
                evaluator = BlockingEvaluator(
                    norm_source=in_ds,
                    raw_source=in_ds,
                    config=b_cfg,
                    logger_instance=logger,
                )
                benchmark_out = evaluator.run_benchmark(
                    strategy_names=["exact_name", "name_token", "composite"],
                    s1_rel_path="train/train_source1.tsv",
                    s2_rel_path="train/train_source2.tsv",
                    s3_rel_path="train/train_source3.tsv",
                    gt_rel_path="train/train_ground_truth.tsv",
                )
                report_data = benchmark_out["report"]
                candidates = benchmark_out["candidates"]

                validator = BlockingValidator(logger_instance=logger)
                val_report = validator.validate(
                    candidates,
                    metrics=report_data["strategy_details"]["composite"],
                    config_dict=b_cfg.to_dict(),
                    input_normalization_version="v001",
                    git_commit="smoke-test",
                )
                assert val_report.status == "PASS"

                saved = save_all_blocking_artifacts(
                    output_source=out_ds,
                    candidate_pairs=candidates,
                    blocking_stats=report_data["strategy_details"]["composite"],
                    strategy_results=report_data,
                    validation_report=val_report,
                    config=b_cfg,
                )
                assert out_ds.exists("candidate_pairs.tsv")
                assert out_ds.exists("blocking_stats.json")
                assert out_ds.exists("blocking_strategy_results.json")
                assert out_ds.exists("blocking_validation_report.json")
                assert out_ds.exists("blocking_metadata.json")
                logger.info("  [PASS] Phase 4 sample candidate blocking pipeline executed and PASSED.")
        except Exception as e:
            logger.error("  [FAIL] Phase 4 candidate blocking error: %s", e)
            failures.append(f"Phase 4 candidate blocking error: {e}")

    if failures:
        logger.error("Smoke test FAILED with %d error(s):", len(failures))
        for err in failures:
            logger.error("  - %s", err)
        return False

    logger.info("Smoke test PASSED! Project infrastructure, Phase 1, Phase 2, Phase 3, and Phase 4 pipelines are operational.")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
