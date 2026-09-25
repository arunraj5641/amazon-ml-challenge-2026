#!/usr/bin/env python3
"""CLI entry point for Phase 3 Training Pair Generation.

Usage examples:
    # Local sample execution:
    python scripts/build_training_pairs.py \
        --raw-input ./tests/fixtures/sample_data \
        --norm-input ./artifacts/normalized/v001 \
        --output ./artifacts/training_pairs/v001

    # SageMaker / S3 execution:
    python scripts/build_training_pairs.py \
        --raw-input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
        --norm-input s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
        --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/training_pairs/v001/ \
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
from src.pairs.builder import TrainingPairBuilder, TrainingPairConfig
from src.pairs.profiler import GroundTruthProfiler
from src.pairs.validator import PairValidator
from src.utils.logger import setup_logger


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Phase 3 Training Pair Generation"
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
        help="Normalized dataset location containing train/train_source2_normalized.tsv (local dir or s3://).",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_uri",
        type=str,
        default=None,
        help="Output location for training pairs (local dir or s3://). Defaults to artifacts/training_pairs/v001.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=1,
        help="Number of negative pairs to sample per positive pair (default: 1).",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.5,
        help="Proportion of negatives from hard (within-country/domain) pool (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Streaming chunksize for reading S3 tables (default: 100000).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default="ap-southeast-2",
        help="AWS region (default: ap-southeast-2).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for test runs.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    logger = setup_logger(name="build_training_pairs")
    logger.info("Initializing Phase 3 Training Pair Builder CLI...")

    config_obj = load_config()

    # Resolve default URIs
    raw_uri = args.raw_input_uri or f"s3://{config_obj.storage.bucket_name}/raw/dataset"
    norm_uri = args.norm_input_uri or f"s3://{config_obj.storage.bucket_name}/artifacts/normalized/v001"
    output_uri = args.output_uri or f"s3://{config_obj.storage.bucket_name}/artifacts/training_pairs/v001"

    logger.info("Configuration:")
    logger.info("  Raw GT Input URI : %s", raw_uri)
    logger.info("  Norm Input URI   : %s", norm_uri)
    logger.info("  Output URI       : %s", output_uri)
    logger.info("  Negatives per Pos: %d", args.negatives_per_positive)
    logger.info("  Hard Neg Ratio   : %.2f", args.hard_negative_ratio)
    logger.info("  Random Seed      : %d", args.seed)

    raw_source = create_data_source(raw_uri, region=args.aws_region)
    norm_source = create_data_source(norm_uri, region=args.aws_region)
    output_source = create_data_source(output_uri, region=args.aws_region)

    # 1. Ground Truth Profiling
    logger.info("Step 1/5: Profiling ground truth...")
    profiler = GroundTruthProfiler(
        data_source=raw_source,
        ground_truth_rel_path="train/train_ground_truth.tsv",
        chunksize=args.chunksize,
        logger=logger,
    )
    profile = profiler.profile()

    # 2. Extract Candidate Pools from normalized sources
    logger.info("Step 2/5: Streaming candidate pools from normalized tables...")
    p_s2, p_s3, c_pools = TrainingPairBuilder.load_candidate_pools(
        norm_source=norm_source,
        chunksize=args.chunksize,
        logger=logger,
    )

    # 3. Build Pairs
    logger.info("Step 3/5: Constructing positive & negative pairs...")
    pair_config = TrainingPairConfig(
        negatives_per_positive=args.negatives_per_positive,
        hard_negative_ratio=args.hard_negative_ratio,
        seed=args.seed,
        chunksize=args.chunksize,
    )
    builder = TrainingPairBuilder(
        input_source=raw_source,
        output_source=output_source,
        config=pair_config,
        logger=logger,
    )

    pairs_df, stats = builder.build_pairs_from_ground_truth(
        gt_rel_path="train/train_ground_truth.tsv",
        candidate_pool_s2=p_s2,
        candidate_pool_s3=p_s3,
        country_candidate_pool=c_pools,
        max_rows=args.max_rows,
    )

    # 4. Validate Pairs
    logger.info("Step 4/5: Validating training pairs...")
    pos_mask = pairs_df["label"] == 1
    auth_positives = set(
        zip(pairs_df.loc[pos_mask, "source1_entity_id"], pairs_df.loc[pos_mask, "target_entity_id"])
    )
    val_result = builder.validator.validate(
        pairs_df=pairs_df,
        authoritative_positives=auth_positives,
        require_all_positives=(args.max_rows is None),
    )

    if val_result.status != "PASS":
        logger.error("Phase 3 validation FAILED with %d errors!", len(val_result.errors))
        for err in val_result.errors[:10]:
            logger.error("  - %s", err)
        sys.exit(1)

    # 5. Persist Artifacts
    logger.info("Step 5/5: Persisting Phase 3 artifacts...")
    saved_uris = builder.save_artifacts(
        pairs_df=pairs_df,
        stats=stats,
        validation_result=val_result,
        ground_truth_profile=profile,
    )

    logger.info("=" * 60)
    logger.info("Phase 3 Training Pair Generation COMPLETE and verified!")
    for k, uri in saved_uris.items():
        logger.info("  %s: %s", k, uri)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
