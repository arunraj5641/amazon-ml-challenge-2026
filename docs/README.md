# Amazon ML Challenge 2026 Documentation

## Overview

This repository contains the infrastructure, pipelines, and models for the Amazon ML Challenge 2026 Entity Resolution project.

## Architecture

The project follows a decoupled, frozen architecture:

```
Mac + Antigravity (Local IDE & Prototyping)
        ↓
GitHub (Source of truth for code, configurations, & tests)
        ↓
Sage / SageMaker (Heavy computation, candidate generation, model training)
        ↓
S3 (Persistent checkpoints, features, and model artifacts)
```

## Phase Roadmap

- **Phase 0: Infrastructure & Project Setup**
  - Clean project layout, configuration, logging, testing harness, and smoke test.
  - No ML logic, no dataset dependencies.
- **Phase 1: Data Ingestion & Validation (Current)**
  - Configurable `DataSource` abstraction (`LocalDataSource` and `S3DataSource`).
  - Validation of the 7 official challenge TSV files.
  - TSV schema validation, data quality checks (null, empty, whitespace counts).
  - Entity ID integrity and uniqueness validation.
  - Ground truth reference validation (Source 1 existence, match existence in Source 2 or 3, 0/1/multi matches).
  - Open-set country statistics.
  - Memory-conscious chunked streaming.
  - CLI runner generating `validation_report.json` and `dataset_stats.json`.
- **Future Phases (2+)**:
  - Phase 2: Text/Attribute normalization and cleaning.
  - Phase 3: Training pair generation.
  - Phase 4: Blocking & candidate filtering.
  - Phase 5: Final candidate generation pipeline.
  - Phase 6: Feature engineering.
  - Phase 7: ML matcher and classification models.
  - Phase 8: Decision engine & threshold calibration.
  - Phase 9+: Error analysis, test inference, and submission generation.

---

## Phase 1: Data Ingestion & Validation Guide

### 1. Official Dataset vs. Local Synthetic Data

- **Official Dataset**: Stored immutably in S3 at:
  ```text
  s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/
  ```
  The full dataset (~2.3+ GB) is **never** downloaded to the local Mac and **never** committed to Git.
- **Local Synthetic Fixtures**: Stored in `tests/fixtures/sample_data/` for rapid unit testing and local development without requiring AWS credentials or network access.

### 2. Expected 7 TSV Input Files

The official dataset contains exactly these 7 TSV files (parsed with `sep="\t"`):

```text
train/train_source1.tsv
train/train_source2.tsv
train/train_source3.tsv
train/train_ground_truth.tsv

test/test_source1.tsv
test/test_source2.tsv
test/test_source3.tsv
```

#### Schemas:

- **Source Files** (`train_source1`, `train_source2`, `train_source3`, `test_source1`, `test_source2`, `test_source3`):
  - `entity_id`
  - `business_name`
  - `business_address`
  - `country`
- **Ground Truth** (`train_ground_truth.tsv`):
  - `source1_entity_id`
  - `matched_entity_ids` (comma-separated string, e.g. `s2_101,s3_101`, or empty/blank for 0 matches)

### 3. Running Data Validation

#### Local Mode (Sample Data)

```bash
python scripts/validate_data.py \
    --input ./tests/fixtures/sample_data \
    --output ./artifacts/validation
```

#### S3 Mode (SageMaker Execution)

```bash
python scripts/validate_data.py \
    --input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
    --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/validation/v001/ \
    --aws-region ap-southeast-2 \
    --chunksize 50000
```

### 4. Output Reports

The validation process produces two JSON artifacts:

1. `validation_report.json`: High-level summary with status (`PASS` or `FAIL`), list of present/missing files, detected errors, warnings, and summary totals.
2. `dataset_stats.json`: Granular machine-readable statistics including:
   - Row counts and column inventories
   - Per-column null, empty-string, and whitespace-only counts
   - Unique vs. duplicate entity ID counts
   - Full open-set country distribution (row counts and percentages)
   - Ground truth metrics (0 matches, single match, multi-match counts, invalid reference counts)

### 5. AWS Permissions & Environment Assumptions

- When running against S3 in SageMaker, the execution role attached to the SageMaker notebook/training job must have `s3:GetObject` and `s3:ListBucket` on `sagemaker-ap-southeast-2-904290466033/raw/dataset/*`, and `s3:PutObject` on the target artifacts prefix.
- Standard `boto3` session credentials and IAM roles are used automatically; no credentials should ever be hardcoded or written to configuration files.

---

## Experiment Tracking

All experimental runs, git hashes, pipeline versions, hyperparameters, and validation metrics are logged in [results.csv](file:///Users/arunraj/amazon-ml-challenge/amazon-ml-challenge-2026/experiments/results.csv).
