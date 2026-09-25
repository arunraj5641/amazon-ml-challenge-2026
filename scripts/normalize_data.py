#!/usr/bin/env python3
"""CLI entry point for Phase 2 Data Normalization.

Usage examples:
    # Local sample normalization:
    python scripts/normalize_data.py \
        --input ./tests/fixtures/sample_data \
        --output ./artifacts/normalized/v001

    # SageMaker / S3 execution:
    python scripts/normalize_data.py \
        --input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
        --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
        --aws-region ap-southeast-2
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.data_source import create_data_source
from src.normalization.pipeline import NormalizationPipeline, NORMALIZATION_VERSION
from src.utils.logger import setup_logger


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Phase 2 Data Normalization"
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
        help="Output location for normalized artifacts (local dir or s3://bucket/prefix). Defaults to artifacts/normalized/v001.",
    )
    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        default=50000,
        help="Chunksize for memory-efficient streaming normalization (default: 50000).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help="AWS region name for S3 operations (default: from config or ap-southeast-2).",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=NORMALIZATION_VERSION,
        help="Normalization artifact version (default: v001).",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    config = load_config()
    logger = setup_logger(name="normalize_cli", level=config.log_level)

    # Resolve input URI
    input_uri = args.input_uri
    if not input_uri:
        if config.storage.enabled and config.storage.bucket_name:
            input_uri = f"s3://{config.storage.bucket_name}/{config.storage.data_prefix}"
        else:
            input_uri = str(config.paths.data_dir)

    # Resolve output URI
    output_uri = args.output_uri
    if not output_uri:
        if input_uri.startswith("s3://") and config.storage.bucket_name:
            output_uri = f"s3://{config.storage.bucket_name}/artifacts/normalized/{args.version}"
        else:
            output_uri = str(config.paths.artifacts_dir / "normalized" / args.version)

    region = args.aws_region or config.storage.region

    logger.info("Initializing Phase 2 Normalization Pipeline:")
    logger.info("  Input Location : %s", input_uri)
    logger.info("  Output Location: %s", output_uri)
    logger.info("  Chunksize      : %d", args.chunksize)
    logger.info("  Version        : %s", args.version)
    if input_uri.startswith("s3://") or output_uri.startswith("s3://"):
        logger.info("  AWS Region     : %s", region)

    try:
        input_source = create_data_source(input_uri, region=region)
        output_source = create_data_source(output_uri, region=region)

        pipeline = NormalizationPipeline(
            input_source=input_source,
            output_source=output_source,
            chunksize=args.chunksize,
            version=args.version,
            logger=logger,
        )

        report = pipeline.run()

        logger.info("--------------------------------------------------")
        logger.info("Normalization Execution Summary:")
        logger.info("  Status         : %s", report["status"])
        logger.info("  Version        : %s", report["version"])
        logger.info("  Files Processed: %d", len(report["files_processed"]))
        logger.info("  Total Rows     : %d", report["total_rows_processed"])
        logger.info("  Duration       : %ss", report["processing_duration_seconds"])
        logger.info("  Errors         : %d", len(report["errors"]))
        logger.info("--------------------------------------------------")

        if report["status"] != "PASS":
            logger.error("Normalization failed with errors:")
            for err in report["errors"][:10]:
                logger.error("  - %s", err)
            sys.exit(1)

        logger.info("Normalization PASSED successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error("Fatal error during normalization: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
