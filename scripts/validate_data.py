#!/usr/bin/env python3
"""CLI entry point for Phase 1 Data Ingestion & Validation.

Usage examples:
    # Local sample validation
    python scripts/validate_data.py --input ./tests/fixtures/sample_data --output ./artifacts/validation

    # S3 execution in SageMaker
    python scripts/validate_data.py \
        --input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
        --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/validation/v001/ \
        --aws-region ap-southeast-2
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path if not already present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.data_source import create_data_source
from src.data.validator import DataValidator
from src.utils.logger import setup_logger


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Phase 1 Data Ingestion & Validation"
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_uri",
        type=str,
        default=None,
        help="Input dataset location (local directory or s3://bucket/prefix). Defaults to config paths/data_dir.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_uri",
        type=str,
        default=None,
        help="Output location for validation reports (local dir or s3://bucket/prefix). Defaults to artifacts/validation.",
    )
    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        default=50000,
        help="Number of rows per chunk for streaming TSV parsing (default: 50000).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help="AWS region name for S3 data source (default: from config or ap-southeast-2).",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    config = load_config()
    logger = setup_logger(name="validate_cli", level=config.log_level)

    # Resolve input
    input_uri = args.input_uri
    if not input_uri:
        if config.storage.enabled and config.storage.bucket_name:
            input_uri = f"s3://{config.storage.bucket_name}/{config.storage.data_prefix}"
        else:
            input_uri = str(config.paths.data_dir)

    # Resolve output
    output_uri = args.output_uri
    if not output_uri:
        if input_uri.startswith("s3://") and config.storage.bucket_name:
            output_uri = f"s3://{config.storage.bucket_name}/{config.storage.artifacts_prefix}"
        else:
            output_uri = str(config.paths.artifacts_dir / "validation")

    region = args.aws_region or config.storage.region

    logger.info("Initializing Phase 1 validation pipeline:")
    logger.info("  Input Location : %s", input_uri)
    logger.info("  Output Location: %s", output_uri)
    logger.info("  Chunksize      : %d", args.chunksize)
    if input_uri.startswith("s3://") or output_uri.startswith("s3://"):
        logger.info("  AWS Region     : %s", region)

    try:
        data_source = create_data_source(input_uri, region=region)
        validator = DataValidator(
            data_source=data_source,
            chunksize=args.chunksize,
            logger=logger,
        )

        result = validator.validate_all()
        rep_uri, stats_uri = validator.write_reports(output_uri, result)

        logger.info("--------------------------------------------------")
        logger.info("Validation Result Summary:")
        logger.info("  Status         : %s", result.status)
        logger.info("  Files Checked  : %d", result.files_checked)
        logger.info("  Files Present  : %d", len(result.files_present))
        logger.info("  Files Missing  : %d", len(result.files_missing))
        logger.info("  Total Errors   : %d", len(result.errors))
        logger.info("  Total Warnings : %d", len(result.warnings))
        logger.info("  Report URI     : %s", rep_uri)
        logger.info("  Stats URI      : %s", stats_uri)
        logger.info("--------------------------------------------------")

        if result.status != "PASS":
            logger.error("Validation failed. Top errors:")
            for err in result.errors[:10]:
                logger.error("  - %s", err)
            if len(result.errors) > 10:
                logger.error("  ... and %d more errors.", len(result.errors) - 10)
            sys.exit(1)

        logger.info("Validation PASSED successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error("Fatal error during validation: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
