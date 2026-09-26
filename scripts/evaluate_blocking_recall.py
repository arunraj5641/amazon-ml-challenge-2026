#!/usr/bin/env python3
"""Standalone Phase 4 Blocking Recall Evaluator CLI.

Evaluates recall of Phase 4 candidate pairs against authoritative Phase 3 ground truth.
Streams both candidate pairs (~500M rows) and missed positive pairs without buffering
all candidates into memory.

Usage:
    python scripts/evaluate_blocking_recall.py \
        --candidate-pairs-uri s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/candidate_pairs.tsv \
        --ground-truth-uri s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/train/train_ground_truth.tsv \
        --output-uri s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001_recall_eval/
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
from src.blocking.recall_evaluator import (
    BlockingRecallEvaluator,
    get_git_commit_hash,
    load_authoritative_positives,
    stream_text_lines,
)
from src.data.data_source import create_data_source
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 Blocking Recall Evaluator on Full Ground Truth."
    )
    parser.add_argument(
        "--candidate-pairs-uri",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/candidate_pairs.tsv",
        help="Path or S3 URI to candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--ground-truth-uri",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/train/train_ground_truth.tsv",
        help="Path or S3 URI to train_ground_truth.tsv or training_pairs.tsv.",
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default="s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001_recall_eval/",
        help="Destination directory or S3 prefix for evaluation artifacts.",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default="ap-southeast-2",
        help="AWS region for S3 data sources.",
    )
    parser.add_argument(
        "--total-s1-pool",
        type=int,
        default=2_206_821,
        help="Total Source 1 entity count for reduction calculation.",
    )
    parser.add_argument(
        "--total-target-pool",
        type=int,
        default=10_320_219,
        help="Total Target pool count (|S2| + |S3|) for reduction calculation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(name="evaluate_blocking_recall")
    logger.info("Initializing Phase 4 Blocking Recall Evaluator CLI...")
    logger.info("Candidate pairs URI : %s", args.candidate_pairs_uri)
    logger.info("Ground truth URI    : %s", args.ground_truth_uri)
    logger.info("Output URI          : %s", args.output_uri)
    logger.info("AWS Region          : %s", args.aws_region)

    output_source = create_data_source(args.output_uri, region=args.aws_region)
    git_hash = get_git_commit_hash()

    # Step 1: Load authoritative positives
    logger.info("Step 1/3: Loading authoritative ground truth...")
    t0 = time.time()
    sorted_gt = load_authoritative_positives(args.ground_truth_uri, region=args.aws_region)
    logger.info("Loaded %d ground-truth positives in %.1fs", len(sorted_gt), time.time() - t0)

    # Step 2: Stream evaluation with O(1) RAM
    logger.info("Step 2/3: Streaming candidate evaluation and generating missed pairs...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_missed_tsv = Path(tmp_dir) / "missed_positive_pairs.tsv.gz"
        evaluator = BlockingRecallEvaluator(logger_instance=logger)
        candidate_lines = stream_text_lines(args.candidate_pairs_uri, region=args.aws_region)

        report = evaluator.evaluate(
            candidate_lines=candidate_lines,
            sorted_ground_truth=sorted_gt,
            missed_output_tsv_path=local_missed_tsv,
            total_s1_pool_size=args.total_s1_pool,
            total_target_pool_size=args.total_target_pool,
        )

        metadata = {
            "evaluation_type": "blocking_recall_evaluation",
            "git_commit": git_hash,
            "inputs": {
                "candidate_pairs_uri": args.candidate_pairs_uri,
                "ground_truth_uri": args.ground_truth_uri,
            },
            "output_destination": output_source.get_uri(),
            "config": {
                "total_s1_pool": args.total_s1_pool,
                "total_target_pool": args.total_target_pool,
                "aws_region": args.aws_region,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_status": report["integrity_validation"]["status"],
        }

        # Step 3: Persist artifacts
        logger.info("Step 3/3: Persisting evaluation artifacts to %s...", output_source.get_uri())
        saved_uris = {}

        # 1. blocking_recall_report.json
        saved_uris["blocking_recall_report_json"] = save_json_artifact(
            report, output_source, "blocking_recall_report.json"
        )

        # 2. missed_positive_pairs.tsv.gz
        if local_missed_tsv.exists():
            saved_uris["missed_positive_pairs_tsv_gz"] = save_candidate_pairs_file(
                local_missed_tsv, output_source, "missed_positive_pairs.tsv.gz"
            )

        # 3. evaluation_metadata.json
        saved_uris["evaluation_metadata_json"] = save_json_artifact(
            metadata, output_source, "evaluation_metadata.json"
        )

        logger.info("=" * 60)
        logger.info("Phase 4 Recall Evaluation COMPLETE!")
        logger.info("Overall Recall : %.6f", report["overall"]["blocking_recall"])
        logger.info("Recovered      : %d", report["overall"]["recovered_positive_pairs"])
        logger.info("Missed         : %d", report["overall"]["missed_positive_pairs"])
        logger.info("Total Positives: %d", report["overall"]["total_positive_pairs"])
        logger.info("Source 2 Recall: %.6f (%d/%d)",
                    report["per_target_source"]["source2"]["recall"],
                    report["per_target_source"]["source2"]["recovered"],
                    report["per_target_source"]["source2"]["total_positives"])
        logger.info("Source 3 Recall: %.6f (%d/%d)",
                    report["per_target_source"]["source3"]["recall"],
                    report["per_target_source"]["source3"]["recovered"],
                    report["per_target_source"]["source3"]["total_positives"])
        logger.info("Artifacts saved:")
        for k, uri in saved_uris.items():
            logger.info("  %s: %s", k, uri)
        logger.info("=" * 60)

        if report["integrity_validation"]["status"] != "PASS":
            logger.error("Integrity validation FAILED with errors:")
            for err in report["integrity_validation"]["errors"]:
                logger.error("  - %s", err)
            sys.exit(1)


if __name__ == "__main__":
    main()
