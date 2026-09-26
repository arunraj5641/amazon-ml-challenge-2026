#!/usr/bin/env python3
"""CLI entry point for Phase 4 Candidate Generation.

Generates candidate target pairs for Source 1 entities:
- Supports composite union or specific blocking strategy families.
- Enforces frequency guards on high-frequency blocking keys.
- Validates candidate pairs and outputs candidate_pairs.tsv and statistics.

Usage examples:
    # Local sample execution:
    python scripts/generate_candidates.py \
        --norm-input ./tests/fixtures/sample_data \
        --output ./artifacts/candidates/block_v001 \
        --strategy composite

    # SageMaker / S3 execution:
    python scripts/generate_candidates.py \
        --norm-input s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
        --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/ \
        --aws-region ap-southeast-2 \
        --strategy composite
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.blocking.config import BLOCKING_VERSION, BlockingConfig
from src.blocking.generator import CandidateGenerator
from src.blocking.io import save_all_blocking_artifacts, save_candidate_pairs, save_json_artifact
from src.blocking.strategies import create_default_strategies, create_strategy
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
        description="Amazon ML Challenge 2026 - Phase 4 Candidate Generation"
    )
    parser.add_argument(
        "--norm-input",
        dest="norm_input_uri",
        type=str,
        default=None,
        help="Normalized dataset location containing source files (local dir or s3://).",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_uri",
        type=str,
        default=None,
        help="Output location for generated candidate pairs. Defaults to artifacts/candidates/block_v001.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="composite",
        help="Blocking strategy name ('composite', 'exact_name', 'name_token', 'name_prefix', 'address_conservative', 'country_scoped_name').",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional deterministic sample size of Source 1 entities for fast execution.",
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
    logger = setup_logger(name="generate_candidates")
    logger.info("Initializing Phase 4 Candidate Generation CLI...")

    project_cfg = load_config()

    norm_uri = args.norm_input_uri or f"s3://{project_cfg.storage.bucket_name}/artifacts/normalized/v001"
    output_uri = args.output_uri or f"s3://{project_cfg.storage.bucket_name}/artifacts/candidates/block_v001"

    logger.info("Configuration:")
    logger.info("  Norm Input URI        : %s", norm_uri)
    logger.info("  Output URI            : %s", output_uri)
    logger.info("  Strategy              : %s", args.strategy)
    logger.info("  Sample Size           : %s", args.sample_size)
    logger.info("  Seed                  : %d", args.seed)
    logger.info("  Max Posting List Size : %d", args.max_posting_list_size)
    logger.info("  Country Agreement     : %s", args.country_agreement_mode)

    norm_source = create_data_source(norm_uri, region=args.aws_region)
    output_source = create_data_source(output_uri, region=args.aws_region)

    blocking_config = BlockingConfig(
        max_posting_list_size=args.max_posting_list_size,
        country_agreement_mode=args.country_agreement_mode,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    available_strats = create_default_strategies(blocking_config)
    if args.strategy in available_strats:
        strategy = available_strats[args.strategy]
    else:
        strategy = create_strategy(args.strategy, blocking_config)

    start_time = time.time()

    # 1. Build Index for Targets
    logger.info("Step 1/4: Building target index using strategy '%s'...", strategy.name)
    index = CandidateGenerator.build_index_from_sources(
        norm_source=norm_source,
        strategy=strategy,
        config=blocking_config,
        logger_instance=logger,
    )

    # 2. Stream Source 1 and Generate Candidates
    logger.info("Step 2/4: Generating candidates for Source 1 entities...")
    generator = CandidateGenerator(strategy=strategy, index=index, config=blocking_config)

    s1_path = "train/train_source1_normalized.tsv"
    if not norm_source.exists(s1_path):
        alt = s1_path.replace("_normalized.tsv", ".tsv")
        if norm_source.exists(alt):
            s1_path = alt

    all_candidate_pairs = []
    s1_processed = 0

    for chunk in norm_source.read_chunks(s1_path, sep="\t", chunksize=args.chunksize):
        recs = chunk.to_dict(orient="records")
        for rec in recs:
            s1_processed += 1
            if args.sample_size and s1_processed > args.sample_size:
                break
            cands = generator.generate_candidates_for_record(rec)
            all_candidate_pairs.extend(cands)
        if args.sample_size and s1_processed >= args.sample_size:
            break

    elapsed = time.time() - start_time
    logger.info(
        "Candidate generation complete: %d Source 1 records -> %d candidate pairs in %.2fs",
        s1_processed,
        len(all_candidate_pairs),
        elapsed,
    )

    # Globally sort deterministically before validation and persistence
    all_candidate_pairs.sort(key=lambda p: (p.source1_entity_id, p.target_entity_id))

    # 3. Validate
    logger.info("Step 3/4: Validating candidate pairs...")
    validator = BlockingValidator(logger_instance=logger)
    git_hash = get_git_commit_hash()
    val_report = validator.validate(
        candidate_pairs=all_candidate_pairs,
        input_normalization_version="v001",
        git_commit=git_hash,
        config_dict=blocking_config.to_dict(),
    )

    if val_report.status != "PASS":
        logger.error("Phase 4 Validation FAILED with %d error(s):", len(val_report.errors))
        for err in val_report.errors:
            logger.error("  - %s", err)
        sys.exit(1)

    # 4. Save Artifacts
    logger.info("Step 4/4: Persisting candidate artifacts...")
    stats = {
        "strategy_name": strategy.name,
        "evaluated_s1_count": s1_processed,
        "total_candidate_pairs": len(all_candidate_pairs),
        "runtime_seconds": round(elapsed, 4),
        "index_stats": index.get_stats(),
    }

    saved_uris = save_all_blocking_artifacts(
        output_source=output_source,
        candidate_pairs=all_candidate_pairs,
        blocking_stats=stats,
        strategy_results={"primary": stats},
        validation_report=val_report,
        config=blocking_config,
        git_commit=git_hash,
        normalization_version="v001",
    )

    logger.info("=" * 60)
    logger.info("Phase 4 Candidate Generation COMPLETE!")
    for k, uri in saved_uris.items():
        logger.info("  %s: %s", k, uri)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
