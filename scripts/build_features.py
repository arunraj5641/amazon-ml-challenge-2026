#!/usr/bin/env python3
"""Phase 6 CLI: Feature Engineering Pipeline.

Generates numerical pairwise features for training pairs and/or final candidate pairs
with streaming memory safety and identical feature schemas.

Usage examples:
    # Generate training features only:
    python scripts/build_features.py --mode train

    # Generate candidate features only:
    python scripts/build_features.py --mode candidates

    # Smoke test on 1,000 rows:
    python scripts/build_features.py --mode all --max-rows 1000
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.blocking.io import save_candidate_pairs_file, save_json_artifact
from src.blocking.recall_evaluator import stream_text_lines
from src.candidates.pipeline import get_git_commit_hash
from src.data.data_source import create_data_source
from src.features.config import FEATURE_NAMES, FEATURE_VERSION, FeatureConfig
from src.features.pipeline import FeaturePipeline
from src.features.schema import build_default_feature_schema
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 6 Feature Engineering CLI for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "candidates", "all"],
        default="all",
        help="Feature generation target: 'train', 'candidates', or 'all'.",
    )
    parser.add_argument(
        "--norm-input",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/",
        help="Directory URI containing normalized source TSVs (train_source1_normalized.tsv, etc.).",
    )
    parser.add_argument(
        "--training-pairs-input",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/training_pairs/v001/training_pairs.tsv",
        help="Path or URI to Phase 3 training_pairs.tsv.",
    )
    parser.add_argument(
        "--candidates-input",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/final_v001/candidate_pairs.tsv",
        help="Path or URI to Phase 5 candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/features/feature_v001/",
        help="Destination directory or S3 prefix for feature artifacts.",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default="ap-southeast-2",
        help="AWS region for S3 data sources.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Streaming chunk size for file buffering.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for smoke testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(name="build_features")
    logger.info("Initializing Phase 6 Feature Engineering CLI...")
    logger.info("Mode                : %s", args.mode)
    logger.info("Normalized Input    : %s", args.norm_input)
    logger.info("Training Pairs Input: %s", args.training_pairs_input)
    logger.info("Candidates Input    : %s", args.candidates_input)
    logger.info("Output URI          : %s", args.output_uri)
    logger.info("AWS Region          : %s", args.aws_region)
    logger.info("Max Rows            : %s", args.max_rows)

    output_source = create_data_source(args.output_uri, region=args.aws_region)
    git_hash = get_git_commit_hash()

    config = FeatureConfig(
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
    pipeline = FeaturePipeline(config=config, logger_instance=logger)

    # 1. Resolve normalized sources to load
    norm_base = args.norm_input.rstrip("/")
    source_files = [
        f"{norm_base}/train/train_source1_normalized.tsv",
        f"{norm_base}/train/train_source2_normalized.tsv",
        f"{norm_base}/train/train_source3_normalized.tsv",
        f"{norm_base}/train_source1_normalized.tsv",
        f"{norm_base}/train_source2_normalized.tsv",
        f"{norm_base}/train_source3_normalized.tsv",
    ]
    pipeline.load_normalized_entities(source_files, region=args.aws_region)

    overall_stats: Dict[str, Any] = {
        "phase": 6,
        "phase_name": "feature_engineering",
        "feature_version": FEATURE_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "execution_mode": args.mode,
        "git_commit": git_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    # Save feature_schema.json at top level
    schema_dict = build_default_feature_schema()
    save_json_artifact(schema_dict, output_source, "feature_schema.json")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        # 2. Training Features
        if args.mode in ("train", "all"):
            logger.info("Processing Training Pairs Feature Generation...")
            train_tmp = tmp_p / "training"
            train_iter = stream_text_lines(args.training_pairs_input, region=args.aws_region)

            train_stats, train_report = pipeline.extract_training_features(
                training_pairs_iter=train_iter,
                output_dir=train_tmp,
                git_commit=git_hash,
                max_rows=args.max_rows,
            )

            # Persist training artifacts
            save_candidate_pairs_file(train_tmp / "X_train.tsv.gz", output_source, "training/X_train.tsv.gz")
            save_candidate_pairs_file(train_tmp / "y_train.tsv.gz", output_source, "training/y_train.tsv.gz")
            save_candidate_pairs_file(train_tmp / "pair_ids_train.tsv.gz", output_source, "training/pair_ids_train.tsv.gz")
            save_json_artifact(train_stats, output_source, "training/training_stats.json")
            save_json_artifact(train_report.to_dict(), output_source, "training/training_validation_report.json")

            train_meta = {
                "dataset": "training",
                "input_source": args.training_pairs_input,
                "row_count": train_stats.get("total_pairs_processed", 0),
                "feature_count": len(FEATURE_NAMES),
                "validation_status": train_report.status,
                "git_commit": git_hash,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_json_artifact(train_meta, output_source, "training/metadata.json")
            overall_stats["datasets"]["training"] = train_stats

            if train_report.status != "PASS":
                logger.error("Training features validation FAILED!")
                sys.exit(1)

        # 3. Candidate Features
        if args.mode in ("candidates", "all"):
            logger.info("Processing Candidate Pairs Feature Generation...")
            cand_tmp = tmp_p / "candidates"
            cand_iter = stream_text_lines(args.candidates_input, region=args.aws_region)

            cand_stats, cand_report = pipeline.extract_candidate_features(
                candidate_pairs_iter=cand_iter,
                output_dir=cand_tmp,
                git_commit=git_hash,
                max_rows=args.max_rows,
            )

            # Persist candidate artifacts
            save_candidate_pairs_file(cand_tmp / "X_candidates.tsv.gz", output_source, "candidates/X_candidates.tsv.gz")
            save_candidate_pairs_file(cand_tmp / "candidate_ids.tsv.gz", output_source, "candidates/candidate_ids.tsv.gz")
            save_json_artifact(cand_stats, output_source, "candidates/candidate_stats.json")
            save_json_artifact(cand_report.to_dict(), output_source, "candidates/candidate_validation_report.json")

            cand_meta = {
                "dataset": "candidates",
                "input_source": args.candidates_input,
                "row_count": cand_stats.get("total_pairs_processed", 0),
                "feature_count": len(FEATURE_NAMES),
                "validation_status": cand_report.status,
                "git_commit": git_hash,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_json_artifact(cand_meta, output_source, "candidates/metadata.json")
            overall_stats["datasets"]["candidates"] = cand_stats

            if cand_report.status != "PASS":
                logger.error("Candidate features validation FAILED!")
                sys.exit(1)

        # Save top-level feature_stats.json and metadata.json
        save_json_artifact(overall_stats, output_source, "feature_stats.json")
        top_meta = {
            "phase": 6,
            "phase_name": "feature_engineering",
            "version": FEATURE_VERSION,
            "feature_count": len(FEATURE_NAMES),
            "output_destination": output_source.get_uri(),
            "git_commit": git_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_status": "PASS",
        }
        save_json_artifact(top_meta, output_source, "metadata.json")

    logger.info("=" * 60)
    logger.info("Phase 6 Feature Engineering COMPLETE!")
    logger.info("Artifacts saved to: %s", output_source.get_uri())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
