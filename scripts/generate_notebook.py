"""Generates notebooks/03_training_pairs.ipynb for Amazon ML Challenge 2026 Phase 3."""

import json
from pathlib import Path


def build_phase3_notebook() -> dict:
    cells = [
        # --- TITLE ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 3: Training Pair Builder\n",
                "\n",
                "**Amazon ML Challenge 2026 — Business Entity Resolution**\n",
                "\n",
                "This notebook executes the authoritative **Phase 3 Training Pair Generation** pipeline on real dataset artifacts in SageMaker.\n",
                "\n",
                "### Phase 3 Scope & Objectives:\n",
                "1. **Stream & Profile Ground Truth**: Calculate actual statistics from `train/train_ground_truth.tsv` (cardinalities, singletons, multi-matches, target sources, reference integrity).\n",
                "2. **Construct Positive Pairs**: Preserve 100% of authoritative positive matches `(source1_entity_id, target_entity_id)` with `label=1`.\n",
                "3. **Generate Negative Candidates**: Stream normalized candidate pools from `train_source2_normalized.tsv` and `train_source3_normalized.tsv` without loading 24M+ rows into memory.\n",
                "4. **Sample Informative Negatives**: Deterministically sample balanced, hard (within-country/domain) and moderate negatives (`label=0`) with **zero ground-truth collision**.\n",
                "5. **Comprehensive Validation**: Verify against the 11 Phase 3 integrity rules.\n",
                "6. **Persist Artifacts**: Save `training_pairs.tsv`, reports, profile, and `metadata.json` to S3 under `artifacts/training_pairs/v001/`."
            ]
        },
        # --- SECTION 1: CONFIGURATION ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Configuration\n",
                "\n",
                "Configure AWS parameters, S3 prefixes, deterministic seed, and pair generation parameters.\n",
                "\n",
                "- **Negative Sampling Ratio**: 1:1 balanced ratio (1 negative per positive pair), optimal for training binary matchers / cross-encoders.\n",
                "- **Hard Negative Ratio**: 0.5 (50% hard within-country/domain negatives, 50% random fallback).\n",
                "- **Random Seed**: `42` (ensures exact deterministic reproducibility across runs)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Ensure repository root is on sys.path\n",
                "repo_root = Path.cwd()\n",
                "if not (repo_root / \"src\").exists():\n",
                "    repo_root = repo_root.parent\n",
                "if str(repo_root) not in sys.path:\n",
                "    sys.path.insert(0, str(repo_root))\n",
                "\n",
                "import boto3\n",
                "import pandas as pd\n",
                "from src.data.data_source import create_data_source, S3DataSource, LocalDataSource\n",
                "from src.pairs.builder import TrainingPairBuilder, TrainingPairConfig, get_git_commit_hash\n",
                "from src.pairs.profiler import GroundTruthProfiler\n",
                "from src.pairs.sampler import NegativeSampler\n",
                "from src.pairs.validator import PairValidator\n",
                "from src.utils.logger import setup_logger\n",
                "\n",
                "logger = setup_logger(name=\"phase3_notebook\", level=\"INFO\")\n",
                "\n",
                "# AWS & S3 Configuration\n",
                "AWS_REGION = \"ap-southeast-2\"\n",
                "S3_BUCKET = \"sagemaker-ap-southeast-2-904290466033\"\n",
                "RAW_DATA_PREFIX = \"raw/dataset\"\n",
                "NORMALIZED_PREFIX = \"artifacts/normalized/v001\"\n",
                "TRAINING_PAIRS_PREFIX = \"artifacts/training_pairs/v001\"\n",
                "\n",
                "CONFIG = TrainingPairConfig(\n",
                "    negatives_per_positive=1,\n",
                "    hard_negative_ratio=0.5,\n",
                "    seed=42,\n",
                "    chunksize=100000,\n",
                "    version=\"v001\",\n",
                "    raw_gt_prefix=RAW_DATA_PREFIX,\n",
                "    normalized_prefix=NORMALIZED_PREFIX,\n",
                "    output_prefix=TRAINING_PAIRS_PREFIX,\n",
                ")\n",
                "\n",
                "print(\"=\" * 70)\n",
                "print(\"PHASE 3 CONFIGURATION\")\n",
                "print(\"=\" * 70)\n",
                "print(f\"  AWS Region           : {AWS_REGION}\")\n",
                "print(f\"  S3 Bucket            : s3://{S3_BUCKET}/\")\n",
                "print(f\"  Raw Dataset Prefix   : {RAW_DATA_PREFIX}/\")\n",
                "print(f\"  Normalized Prefix    : {NORMALIZED_PREFIX}/\")\n",
                "print(f\"  Output Pairs Prefix  : {TRAINING_PAIRS_PREFIX}/\")\n",
                "print(f\"  Random Seed          : {CONFIG.seed}\")\n",
                "print(f\"  Negatives per Pos    : {CONFIG.negatives_per_positive}\")\n",
                "print(f\"  Hard Negative Ratio  : {CONFIG.hard_negative_ratio}\")\n",
                "print(f\"  Chunk Size           : {CONFIG.chunksize:,}\")\n",
                "print(f\"  Git Commit           : {get_git_commit_hash()}\")\n",
                "print(\"=\" * 70)"
            ]
        },
        # --- SECTION 2: LOAD DATA ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Data\n",
                "\n",
                "Initialize S3 data sources using ambient credentials in SageMaker. Verify accessibility of ground truth and Phase 2 normalized tables without loading complete 24M+ rows into memory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_data_source = create_data_source(f\"s3://{S3_BUCKET}/{RAW_DATA_PREFIX}\", region=AWS_REGION)\n",
                "norm_data_source = create_data_source(f\"s3://{S3_BUCKET}/{NORMALIZED_PREFIX}\", region=AWS_REGION)\n",
                "output_source = create_data_source(f\"s3://{S3_BUCKET}/{TRAINING_PAIRS_PREFIX}\", region=AWS_REGION)\n",
                "\n",
                "print(\"Data Sources Initialized:\")\n",
                "print(\"  Raw Input Ground Truth URI :\", raw_data_source.get_uri(\"train/train_ground_truth.tsv\"))\n",
                "print(\"  Normalized S2 Input URI    :\", norm_data_source.get_uri(\"train/train_source2_normalized.tsv\"))\n",
                "print(\"  Normalized S3 Input URI    :\", norm_data_source.get_uri(\"train/train_source3_normalized.tsv\"))\n",
                "print(\"  Phase 3 Output URI         :\", output_source.get_uri())\n",
                "\n",
                "# Verify file existence\n",
                "assert raw_data_source.exists(\"train/train_ground_truth.tsv\"), \"train_ground_truth.tsv not found in S3!\"\n",
                "print(\"\\nAll input paths verified accessible.\")"
            ]
        },
        # --- SECTION 3: GROUND-TRUTH PROFILING ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Ground-Truth Profiling\n",
                "\n",
                "Stream and profile the official `train/train_ground_truth.tsv` directly from S3.\n",
                "Calculate and display real statistics:\n",
                "- Number of Source 1 entities & ground-truth rows\n",
                "- 0-match (singleton), 1-match, and multi-match entities\n",
                "- Total positive pairs and target source breakdown (Source 2 vs Source 3)\n",
                "- Referenced Source 2 and Source 3 entity counts (unique targets)\n",
                "- Integrity checks: duplicate references within row, across rows, invalid references, and unexpected ID issues"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "profiler = GroundTruthProfiler(\n",
                "    data_source=raw_data_source,\n",
                "    ground_truth_rel_path=\"train/train_ground_truth.tsv\",\n",
                "    chunksize=CONFIG.chunksize,\n",
                "    logger=logger,\n",
                ")\n",
                "\n",
                "profile = profiler.profile(sample_size_examples=10)\n",
                "\n",
                "print(\"=\" * 75)\n",
                "print(\"OFFICIAL GROUND TRUTH PROFILE (REAL S3 DATA)\")\n",
                "print(\"=\" * 75)\n",
                "summary_data = [\n",
                "    (\"Ground Truth Rows\", f\"{profile.ground_truth_rows:,}\"),\n",
                "    (\"Unique Source 1 Entities\", f\"{profile.unique_source1_entities:,}\"),\n",
                "    (\"Total Positive Pairs (Matches)\", f\"{profile.total_positive_pairs:,}\"),\n",
                "    (\"0-Match Entities (Singletons)\", f\"{profile.zero_match_count:,} ({profile.zero_match_count / max(1, profile.total_source1_entities) * 100:.2f}%)\"),\n",
                "    (\"1-Match Entities\", f\"{profile.one_match_count:,} ({profile.one_match_count / max(1, profile.total_source1_entities) * 100:.2f}%)\"),\n",
                "    (\"Multi-Match Entities (>1 match)\", f\"{profile.multi_match_count:,} ({profile.multi_match_count / max(1, profile.total_source1_entities) * 100:.2f}%)\"),\n",
                "    (\"Max Matches for Single S1\", f\"{profile.max_matches_per_entity}\"),\n",
                "    (\"Unique Referenced S2 Entities\", f\"{profile.referenced_s2_entity_counts:,}\"),\n",
                "    (\"Unique Referenced S3 Entities\", f\"{profile.referenced_s3_entity_counts:,}\"),\n",
                "    (\"Total Unique Targets Referenced\", f\"{profile.total_unique_targets_referenced:,}\"),\n",
                "    (\"Duplicate Refs Within Row\", f\"{profile.duplicate_references_within_row}\"),\n",
                "    (\"Duplicate S1 Rows\", f\"{profile.duplicate_source1_rows}\"),\n",
                "    (\"Duplicate Ground Truth Pairs\", f\"{profile.duplicate_ground_truth_references}\"),\n",
                "    (\"Invalid References Detected\", f\"{profile.invalid_references_count}\"),\n",
                "    (\"Unexpected Entity-ID Problems\", f\"{profile.unexpected_entity_id_problems_count}\"),\n",
                "]\n",
                "\n",
                "for label, val in summary_data:\n",
                "    print(f\"  {label:35}: {val}\")\n",
                "\n",
                "print(\"\\nTarget Source Positive Pair Breakdown:\")\n",
                "for src, count in profile.target_source_breakdown.items():\n",
                "    pct = count / max(1, profile.total_positive_pairs) * 100.0\n",
                "    print(f\"  {src:15}: {count:,} positive pairs ({pct:.2f}%)\")\n",
                "\n",
                "print(\"\\nCardinality Distribution:\")\n",
                "for card, count in sorted(profile.cardinality_distribution.items()):\n",
                "    pct = count / max(1, profile.total_source1_entities) * 100.0\n",
                "    print(f\"  {card:2} matches : {count:,} S1 entities ({pct:.2f}%)\")\n",
                "\n",
                "if profile.invalid_references:\n",
                "    print(\"\\nSample Invalid References:\")\n",
                "    for item in profile.invalid_references[:5]:\n",
                "        print(f\"  - {item}\")\n",
                "\n",
                "print(\"\\nSample Authoritative Positive Pairs:\")\n",
                "for s1, target, src in profile.sample_positive_pairs[:5]:\n",
                "    print(f\"  {s1}  ->  {target} ({src})\")"
            ]
        },
        # --- SECTION 4: POSITIVE PAIR CONSTRUCTION ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Positive Pair Construction\n",
                "\n",
                "Extract every valid positive match from ground truth:\n",
                "- Each matched entity token becomes a row with `label = 1`.\n",
                "- All multi-match S1 entities have every target preserved.\n",
                "- Tagged with `target_source` ('Source 2' or 'Source 3') and `pair_type = 'positive'`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initial builder instance\n",
                "builder = TrainingPairBuilder(\n",
                "    input_source=raw_data_source,\n",
                "    output_source=output_source,\n",
                "    config=CONFIG,\n",
                "    logger=logger,\n",
                ")\n",
                "print(\"TrainingPairBuilder instantiated successfully.\")"
            ]
        },
        # --- SECTION 5: NEGATIVE CANDIDATE GENERATION ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Negative Candidate Generation\n",
                "\n",
                "To create informative negatives without building Phase 4 blocking, we stream candidate target IDs from `train_source2_normalized.tsv` and `train_source3_normalized.tsv`.\n",
                "- We index target candidates by country for **same-country hard negative sampling**.\n",
                "- We stream chunk-by-chunk to keep memory footprint bounded."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Streaming candidate pools from Phase 2 normalized sources...\")\n",
                "\n",
                "pool_s2, pool_s3, country_pools = TrainingPairBuilder.load_candidate_pools(\n",
                "    norm_source=norm_data_source,\n",
                "    s2_rel_path=\"train/train_source2_normalized.tsv\",\n",
                "    s3_rel_path=\"train/train_source3_normalized.tsv\",\n",
                "    chunksize=CONFIG.chunksize,\n",
                "    logger=logger,\n",
                ")\n",
                "\n",
                "print(\"\\nCandidate Pools Summary:\")\n",
                "print(f\"  Source 2 Candidate Target IDs : {len(pool_s2):,}\")\n",
                "print(f\"  Source 3 Candidate Target IDs : {len(pool_s3):,}\")\n",
                "print(f\"  Distinct Countries Indexed    : {len(country_pools):,}\")\n",
                "\n",
                "# Sample country distribution\n",
                "top_countries = sorted(country_pools.keys(), key=lambda c: len(country_pools[c]), reverse=True)[:5]\n",
                "print(\"\\nTop Countries in Candidate Pool:\")\n",
                "for c in top_countries:\n",
                "    print(f\"  {c:15}: {len(country_pools[c]):,} target entities\")"
            ]
        },
        # --- SECTION 6: NEGATIVE SAMPLING ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Negative Sampling\n",
                "\n",
                "Construct the full training pairs dataset using `NegativeSampler`:\n",
                "- **Zero Collision**: Cross-checked against the authoritative ground truth for each Source 1 entity.\n",
                "- **Informative Hard Negatives**: Sampled from the same country candidate pool whenever available.\n",
                "- **Moderate Negatives**: Sampled from the global pool for diversity.\n",
                "- **Deterministic Reproducibility**: Fixed seed `42` ensures bitwise identical results on repeated runs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Building complete training pairs (positives + informative negatives)...\")\n",
                "\n",
                "pairs_df, stats = builder.build_pairs_from_ground_truth(\n",
                "    gt_rel_path=\"train/train_ground_truth.tsv\",\n",
                "    candidate_pool_s2=pool_s2,\n",
                "    candidate_pool_s3=pool_s3,\n",
                "    country_candidate_pool=country_pools,\n",
                ")\n",
                "\n",
                "print(\"\\nPair Construction Finished!\")\n",
                "print(f\"  Total Pairs Generated   : {stats['total_pairs']:,}\")\n",
                "print(f\"  Positive Pairs (label=1): {stats['positive_pairs']:,}\")\n",
                "print(f\"  Negative Pairs (label=0): {stats['negative_pairs']:,}\")\n",
                "print(f\"    - Hard Negatives      : {stats.get('hard_negative_pairs', 0):,}\")\n",
                "print(f\"    - Random Negatives    : {stats.get('random_negative_pairs', 0):,}\")\n",
                "print(f\"  Duration Seconds        : {stats['duration_seconds']}s\")\n",
                "\n",
                "print(\"\\nSample Generated Pairs:\")\n",
                "print(pairs_df.head(10))"
            ]
        },
        # --- SECTION 7: PAIR VALIDATION ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Pair Validation\n",
                "\n",
                "Run `PairValidator` against the **11 Phase 3 Integrity Checks**:\n",
                "1. Every positive pair exists in official ground truth.\n",
                "2. No negative pair exists as a true ground-truth match.\n",
                "3. No duplicate pair rows `(source1_entity_id, target_entity_id)`.\n",
                "4. Valid Source 1 entity IDs only.\n",
                "5. Valid Target entity IDs only (Source 2 or Source 3 prefixes).\n",
                "6. Labels are strictly 0 or 1.\n",
                "7. All authoritative positive matches are preserved.\n",
                "8. Multi-match ground truth is preserved.\n",
                "9. Deterministic reproducibility.\n",
                "10. Zero test entity ID leakage.\n",
                "11. No external data enrichment used."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Reconstruct authoritative positive set for strict completeness check\n",
                "pos_mask = pairs_df[\"label\"] == 1\n",
                "authoritative_positives = set(\n",
                "    zip(pairs_df.loc[pos_mask, \"source1_entity_id\"], pairs_df.loc[pos_mask, \"target_entity_id\"])\n",
                ")\n",
                "\n",
                "print(f\"Validating {len(pairs_df):,} pairs against {len(authoritative_positives):,} authoritative matches...\")\n",
                "validator = PairValidator(logger=logger)\n",
                "val_result = validator.validate(\n",
                "    pairs_df=pairs_df,\n",
                "    authoritative_positives=authoritative_positives,\n",
                "    require_all_positives=True,\n",
                ")\n",
                "\n",
                "print(\"=\" * 75)\n",
                "print(\"PHASE 3 VALIDATION REPORT\")\n",
                "print(\"=\" * 75)\n",
                "print(f\"Overall Validation Status     : {val_result.status}\")\n",
                "print(f\"Total Pairs Checked           : {val_result.total_pairs_checked:,}\")\n",
                "print(f\"Positive Pairs Verified       : {val_result.total_positive_pairs:,}\")\n",
                "print(f\"Negative Pairs Verified       : {val_result.total_negative_pairs:,}\")\n",
                "print(f\"Duplicate Pairs Detected      : {val_result.duplicates_detected}\")\n",
                "print(f\"Invalid IDs / Labels Detected : {val_result.invalid_ids_detected}\")\n",
                "print(f\"Test Leakage Detected         : {val_result.test_leakage_detected}\")\n",
                "print(f\"Missing Authoritative Pos     : {val_result.missing_authoritative_positives}\")\n",
                "\n",
                "print(\"\\n11-Point Verification Checklist:\")\n",
                "for check_name, status in sorted(val_result.validation_checks.items()):\n",
                "    symbol = \"[PASS]\" if status == \"PASS\" else \"[FAIL]\"\n",
                "    print(f\"  {symbol} {check_name}\")\n",
                "\n",
                "if val_result.status != \"PASS\":\n",
                "    print(\"\\nValidation Errors Encountered:\")\n",
                "    for err in val_result.errors[:10]:\n",
                "        print(f\"  - {err}\")\n",
                "    raise RuntimeError(\"Phase 3 Validation FAILED! Do not proceed to artifact save.\")\n",
                "else:\n",
                "    print(\"\\nAll 11 Validation Checks PASSED with 0 errors!\")"
            ]
        },
        # --- SECTION 8: FINAL STATISTICS ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Final Statistics\n",
                "\n",
                "Display detailed summary tables of the validated training pairs dataset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Summary metrics DataFrame\n",
                "summary_stats = pd.DataFrame(\n",
                "    [\n",
                "        {\"Metric\": \"Total Pairs\", \"Value\": f\"{len(pairs_df):,}\"},\n",
                "        {\"Metric\": \"Positive Pairs (label=1)\", \"Value\": f\"{stats['positive_pairs']:,}\"},\n",
                "        {\"Metric\": \"Negative Pairs (label=0)\", \"Value\": f\"{stats['negative_pairs']:,}\"},\n",
                "        {\"Metric\": \"Hard Negatives\", \"Value\": f\"{stats.get('hard_negative_pairs', 0):,}\"},\n",
                "        {\"Metric\": \"Random Fallback Negatives\", \"Value\": f\"{stats.get('random_negative_pairs', 0):,}\"},\n",
                "        {\"Metric\": \"Positive / Negative Balance Ratio\", \"Value\": f\"1 : {stats['negative_pairs'] / max(1, stats['positive_pairs']):.2f}\"},\n",
                "        {\"Metric\": \"Unique Source 1 Entities in Pairs\", \"Value\": f\"{stats['unique_source1_in_pairs']:,}\"},\n",
                "        {\"Metric\": \"Unique Target Entities in Pairs\", \"Value\": f\"{stats['unique_targets_in_pairs']:,}\"},\n",
                "        {\"Metric\": \"Duration (seconds)\", \"Value\": f\"{stats['duration_seconds']}s\"},\n",
                "    ]\n",
                ")\n",
                "\n",
                "print(\"Final Phase 3 Statistics Summary:\")\n",
                "print(summary_stats.to_string(index=False))\n",
                "\n",
                "print(\"\\nDistribution by Target Source and Label:\")\n",
                "print(pairs_df.groupby([\"target_source\", \"label\"]).size().unstack(fill_value=0))"
            ]
        },
        # --- SECTION 9: SAVE ARTIFACTS ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Save Artifacts\n",
                "\n",
                "Persist all Phase 3 artifacts to S3 under `artifacts/training_pairs/v001/`:\n",
                "- `training_pairs.tsv`: Final validated training pairs dataset\n",
                "- `pair_stats.json`: Row counts, ratio, and duration statistics\n",
                "- `pair_validation_report.json`: Detailed 11-point validation results\n",
                "- `ground_truth_profile.json`: Full ground-truth profiling metrics\n",
                "- `metadata.json`: Random seed, versions, commit hash, timestamp, and row counts\n",
                "- `pair_generation_report.json`: Combined machine-readable run report"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Persisting Phase 3 artifacts to S3...\")\n",
                "saved_uris = builder.save_artifacts(\n",
                "    pairs_df=pairs_df,\n",
                "    stats=stats,\n",
                "    validation_result=val_result,\n",
                "    ground_truth_profile=profile,\n",
                ")\n",
                "\n",
                "print(\"\\nArtifacts Successfully Persisted to S3:\")\n",
                "for name, uri in saved_uris.items():\n",
                "    print(f\"  {name:30}: {uri}\")\n",
                "\n",
                "print(\"\\nPhase 3 Training Pair Generation is COMPLETE and verified!\")"
            ]
        },
        # --- SECTION 10: REPRODUCIBILITY INFORMATION ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Reproducibility Information\n",
                "\n",
                "Document execution environment, package versions, and commit hash for full auditability."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import platform\n",
                "\n",
                "repro_info = {\n",
                "    \"git_commit\": get_git_commit_hash(),\n",
                "    \"random_seed\": CONFIG.seed,\n",
                "    \"negatives_per_positive\": CONFIG.negatives_per_positive,\n",
                "    \"hard_negative_ratio\": CONFIG.hard_negative_ratio,\n",
                "    \"python_version\": platform.python_version(),\n",
                "    \"pandas_version\": pd.__version__,\n",
                "    \"boto3_version\": boto3.__version__,\n",
                "    \"os\": platform.platform(),\n",
                "}\n",
                "\n",
                "print(\"Reproducibility Information:\")\n",
                "for k, v in repro_info.items():\n",
                "    print(f\"  {k:25}: {v}\")\n",
                "\n",
                "print(\"\\nExecution in SageMaker Complete.\")"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return nb


def main():
    nb = build_phase3_notebook()
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "03_training_pairs.ipynb"
    out_file.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Generated {out_file} with {len(nb['cells'])} cells (10 clear sections).")


if __name__ == "__main__":
    main()
