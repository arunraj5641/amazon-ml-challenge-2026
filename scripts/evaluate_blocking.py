#!/usr/bin/env python3
"""CLI entry point for Phase 4 Blocking Strategy Evaluation & Benchmarking.

Evaluates candidate generation strategies against training ground truth:
- Measures candidate recall, reduction ratio, candidate volume, and runtime.
- Evaluates individual strategy families independently and in composite combinations.
- Enforces frequency guards on high-frequency blocking keys.
- Validates integrity and records comprehensive metadata.

Usage examples:
    # Local sample evaluation:
    python scripts/evaluate_blocking.py \
        --raw-input ./tests/fixtures/sample_data \
        --norm-input ./tests/fixtures/sample_data \
        --output ./artifacts/candidates/block_v001 \
        --sample-size 1000

    # SageMaker / S3 execution:
    python scripts/evaluate_blocking.py \
        --raw-input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
        --norm-input s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
        --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/ \
        --aws-region ap-southeast-2 \
        --sample-size 5000
"""

import argparse
from pathlib import Path
import subprocess
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.blocking.config import BLOCKING_VERSION, BlockingConfig
from src.blocking.evaluator import BlockingEvaluator
from src.blocking.io import save_all_blocking_artifacts
from src.blocking.validator import BlockingValidator
from src.config import load_config
from src.data.data_source import create_data_source
from src.utils.logger import setup_logger


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "phase4-initial-commit"


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Phase 4 Blocking Strategy Evaluation"
    )
    parser.add_argument(
        "--raw-input",
        dest="raw_input_uri",
        type=str,
        default=None,
        help="Raw dataset location containing train/train_ground_truth.tsv (local dir or s3://).",
    )
    parser.add_argument(
        "--norm-input",
        dest="norm_input_uri",
        type=str,
        default=None,
        help="Normalized dataset location containing normalized source files (local dir or s3://).",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_uri",
        type=str,
        default=None,
        help="Output location for blocking evaluation artifacts. Defaults to artifacts/candidates/block_v001.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional deterministic sample size of Source 1 entities for fast benchmarking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--max-posting-list-size",
        type=int,
        default=500,
        help="Maximum posting list size before key is suppressed by frequency guard (default: 500).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated strategy list to evaluate (e.g. 'exact_name,name_token,composite') or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--country-agreement-mode",
        type=str,
        default="allow_missing",
        choices=["allow_missing", "strict", "none"],
        help="Country agreement filter mode: allow_missing (default), strict, or none.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Streaming chunksize for reading tables (default: 50000).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default="ap-southeast-2",
        help="AWS region (default: ap-southeast-2).",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    logger = setup_logger(name="evaluate_blocking")
    logger.info("Initializing Phase 4 Blocking Strategy Evaluation CLI...")

    project_cfg = load_config()

    raw_uri = args.raw_input_uri or f"s3://{project_cfg.storage.bucket_name}/raw/dataset"
    norm_uri = args.norm_input_uri or f"s3://{project_cfg.storage.bucket_name}/artifacts/normalized/v001"
    output_uri = args.output_uri or f"s3://{project_cfg.storage.bucket_name}/artifacts/candidates/block_v001"

    logger.info("Configuration:")
    logger.info("  Raw Input URI        : %s", raw_uri)
    logger.info("  Norm Input URI       : %s", norm_uri)
    logger.info("  Output URI           : %s", output_uri)
    logger.info("  Sample Size          : %s", args.sample_size)
    logger.info("  Seed                 : %d", args.seed)
    logger.info("  Max Posting List Size: %d", args.max_posting_list_size)
    logger.info("  Country Agreement    : %s", args.country_agreement_mode)

    raw_source = create_data_source(raw_uri, region=args.aws_region)
    norm_source = create_data_source(norm_uri, region=args.aws_region)
    output_source = create_data_source(output_uri, region=args.aws_region)

    strat_list = (
        ["exact_name", "name_token", "name_prefix", "address_conservative", "country_scoped_name"]
        if args.strategies.strip().lower() == "all"
        else [s.strip() for s in args.strategies.split(",") if s.strip()]
    )

    blocking_config = BlockingConfig(
        strategies=strat_list,
        max_posting_list_size=args.max_posting_list_size,
        country_agreement_mode=args.country_agreement_mode,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    evaluator = BlockingEvaluator(
        norm_source=norm_source,
        raw_source=raw_source,
        config=blocking_config,
        logger_instance=logger,
    )

    # Run comparative benchmark
    logger.info("Step 1/3: Running strategy benchmark...")
    benchmark_out = evaluator.run_benchmark(
        strategy_names=strat_list,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    report_data = benchmark_out["report"]
    candidate_pairs = benchmark_out["candidates"]
    strat_results = benchmark_out["benchmark_results"]

    # Select primary stats (composite strategy or first available)
    primary_stats = strat_results.get("composite", list(strat_results.values())[0])

    # Validate results
    logger.info("Step 2/3: Validating candidate pairs and metric integrity...")
    validator = BlockingValidator(logger_instance=logger)
    git_hash = get_git_commit_hash()
    val_report = validator.validate(
        candidate_pairs=candidate_pairs,
        metrics=primary_stats,
        input_normalization_version="v001",
        git_commit=git_hash,
        config_dict=blocking_config.to_dict(),
    )

    if val_report.status != "PASS":
        logger.error("Phase 4 Validation FAILED with %d error(s):", len(val_report.errors))
        for err in val_report.errors:
            logger.error("  - %s", err)
        sys.exit(1)

    # Persist artifacts
    logger.info("Step 3/3: Persisting Phase 4 artifacts...")
    saved_uris = save_all_blocking_artifacts(
        output_source=output_source,
        candidate_pairs=candidate_pairs,
        blocking_stats=primary_stats,
        strategy_results=report_data,
        validation_report=val_report,
        config=blocking_config,
        git_commit=git_hash,
        normalization_version="v001",
    )

    logger.info("=" * 60)
    logger.info("Phase 4 Blocking Strategy Evaluation COMPLETE!")
    logger.info("Primary Strategy : %s", primary_stats.get("strategy_name"))
    logger.info("Recall           : %.4f", primary_stats.get("blocking_recall", 0.0))
    logger.info("Reduction Ratio  : %.6f", primary_stats.get("candidate_reduction_ratio", 0.0))
    logger.info("Candidate Pairs  : %d", primary_stats.get("total_candidate_pairs", 0))
    logger.info("Artifacts saved:")
    for k, uri in saved_uris.items():
        logger.info("  %s: %s", k, uri)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
