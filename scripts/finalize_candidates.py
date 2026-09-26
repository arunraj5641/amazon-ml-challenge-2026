#!/usr/bin/env python3
"""Phase 5 CLI: Finalize Candidate Universe for Phase 6 Feature Engineering.

Validates, cleans, and finalizes the ~500M Phase 4 candidate universe with O(1) RAM.
Emits finalized candidate_pairs.tsv, stats, validation report, and metadata.

Usage:
    python scripts/finalize_candidates.py \
        --input-uri s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/ \
        --output-uri s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/final_v001/ \
        --aws-region ap-southeast-2
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import tempfile
import time

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.blocking.io import save_candidate_pairs_file, save_json_artifact
from src.blocking.recall_evaluator import stream_text_lines
from src.candidates.config import (
    BASELINE_CANDIDATE_PAIRS,
    BASELINE_S1_RECORDS,
    BASELINE_TARGET_POOL,
    CandidatePipelineConfig,
)
from src.candidates.pipeline import CandidatePipeline, get_git_commit_hash
from src.data.data_source import create_data_source
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5 Candidate Finalization and Validation CLI."
    )
    parser.add_argument(
        "--input-uri",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/",
        help="Input candidate pairs directory or file URI (defaults to block_v001).",
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/final_v001/",
        help="Destination directory or S3 prefix for finalized candidate artifacts.",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default="ap-southeast-2",
        help="AWS region for S3 data sources.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=BASELINE_CANDIDATE_PAIRS,
        help="Expected candidate count (defaults to Phase 4 baseline 499,912,583).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(name="finalize_candidates")
    logger.info("Initializing Phase 5 Candidate Finalization CLI...")
    logger.info("Input URI      : %s", args.input_uri)
    logger.info("Output URI     : %s", args.output_uri)
    logger.info("AWS Region     : %s", args.aws_region)
    logger.info("Expected Count : %s", f"{args.expected_count:,}" if args.expected_count else "None")

    input_path = args.input_uri.strip()
    if not input_path.endswith(".tsv") and not input_path.endswith(".tsv.gz"):
        input_path = input_path.rstrip("/") + "/candidate_pairs.tsv"

    output_source = create_data_source(args.output_uri, region=args.aws_region)
    git_hash = get_git_commit_hash()

    config = CandidatePipelineConfig(
        expected_candidate_count=args.expected_count,
    )
    pipeline = CandidatePipeline(config=config, logger_instance=logger)

    logger.info("Step 1/2: Streaming candidate pairs through Phase 5 pipeline...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_output_tsv = Path(tmp_dir) / "candidate_pairs.tsv"
        input_lines = stream_text_lines(input_path, region=args.aws_region)

        stats, val_report = pipeline.finalize(
            input_lines=input_lines,
            output_file_path=local_output_tsv,
            git_commit=git_hash,
        )

        metadata = {
            "phase": 5,
            "phase_name": "candidate_pipeline_finalization",
            "candidate_version": config.version,
            "input_blocking_version": config.input_blocking_version,
            "input_source_uri": input_path,
            "output_destination_uri": output_source.get_uri(),
            "git_commit": git_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_status": val_report.status,
            "config": config.to_dict(),
            "summary_metrics": {
                "total_input_candidates": stats["total_input_candidates"],
                "total_output_candidates": stats["total_output_candidates"],
                "unique_s1_entities": stats["unique_s1_entities"],
                "duplicate_candidates_removed": stats["duplicate_candidates_removed"],
            },
        }

        logger.info("Step 2/2: Persisting finalized Phase 5 artifacts to %s...", output_source.get_uri())
        saved_uris = {}

        # 1. candidate_pairs.tsv
        if local_output_tsv.exists():
            saved_uris["candidate_pairs_tsv"] = save_candidate_pairs_file(
                local_output_tsv, output_source, "candidate_pairs.tsv"
            )

        # 2. candidate_stats.json
        saved_uris["candidate_stats_json"] = save_json_artifact(
            stats, output_source, "candidate_stats.json"
        )

        # 3. candidate_validation_report.json
        saved_uris["candidate_validation_report_json"] = save_json_artifact(
            val_report.to_dict(), output_source, "candidate_validation_report.json"
        )

        # 4. metadata.json
        saved_uris["metadata_json"] = save_json_artifact(
            metadata, output_source, "metadata.json"
        )

        logger.info("=" * 60)
        logger.info("Phase 5 Candidate Finalization COMPLETE!")
        logger.info("Status             : %s", val_report.status)
        logger.info("Input Candidates   : %d", stats["total_input_candidates"])
        logger.info("Output Candidates  : %d", stats["total_output_candidates"])
        logger.info("Unique S1 Entities : %d", stats["unique_s1_entities"])
        logger.info("Duplicates Removed : %d", stats["duplicate_candidates_removed"])
        logger.info("Saved Artifacts:")
        for k, uri in saved_uris.items():
            logger.info("  %s: %s", k, uri)
        logger.info("=" * 60)

        if val_report.status != "PASS":
            logger.error("Phase 5 Validation FAILED with errors:")
            for err in val_report.errors:
                logger.error("  - %s", err)
            sys.exit(1)


if __name__ == "__main__":
    main()
