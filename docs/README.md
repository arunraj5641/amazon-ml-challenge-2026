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
- **Phase 3: Training Pair Builder**
  - Profiling ground truth match distributions (singletons, multi-matches).
  - Preserving 100% of authoritative positive pairs.
  - Reproducible hard/moderate negative sampling.
  - Pair validation and artifact persistence.
- **Phase 4: Candidate Blocking & Filtering (Current)**
  - Modular blocking strategies (`exact_name`, `name_token`, `name_prefix`, `address_conservative`, `country_scoped_name`, `composite`).
  - Memory-conscious inverted index with high-frequency key suppression (`max_posting_list_size`).
  - Country-aware blocking supporting open-set countries (including France).
  - Ground truth candidate recall and reduction ratio evaluation.
  - Full details in [docs/PHASE4_BLOCKING.md](PHASE4_BLOCKING.md).
- **Future Phases (5+)**:
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

## Phase 2: Data Normalization Guide

> **Core Principle**: *Raw data is immutable; normalization creates derived artifacts.*

### 1. Purpose of Normalization

In business entity resolution, heterogeneous data sources introduce superficial variations in legal suffix formatting, punctuation, casing, spacing, and country representations (e.g. `ABC Pvt. Ltd.` vs. `ABC PRIVATE LIMITED`). Phase 2 produces deterministic, canonical representations to enable high-recall candidate generation, blocking, and feature engineering in subsequent phases—**while always preserving the raw original fields and entity IDs untouched**.

### 2. Derived Fields Produced

For each of the 6 source TSVs (`train_source1-3`, `test_source1-3`), the pipeline adds 6 normalized columns alongside the 4 original columns:

| Column Name | Category | Description | Example |
| :--- | :--- | :--- | :--- |
| `entity_id` | Original | Preserved verbatim (unique identifier) | `s1_101` |
| `business_name` | Original | Raw business name | `Acme Electronics Corp` |
| `business_name_normalized` | Derived | NFKC Unicode, lowercase, punctuation normalized, ampersands expanded | `acme electronics corp` |
| `business_name_core` | Derived | Normalized name with recognized legal suffixes cleanly removed from end | `acme electronics` |
| `business_name_alnum` | Derived | Alphanumeric tokens only (space separated) for fast indexing/blocking | `acme electronics corp` |
| `business_address` | Original | Raw business address | `100 Market St, San Francisco, CA` |
| `business_address_normalized` | Derived | NFKC Unicode, lowercase, separator punctuation normalized, hyphens preserved | `100 market st san francisco ca` |
| `business_address_alnum` | Derived | Alphanumeric tokens only (space separated) | `100 market st san francisco ca` |
| `country` | Original | Raw country string | `US` |
| `country_normalized` | Derived | Canonical 2-letter ISO code or cleaned lowercase string | `us` |

*Note: Ground truth (`train_ground_truth.tsv`) is immutable and is NOT modified or normalized.*

### 3. Normalization Policies

#### A. Legal Suffix Policy (`business_name_core`)
- Multi-word and single-word legal suffixes (e.g., `private limited`, `pvt ltd`, `inc`, `corp`, `llc`, `gmbh`, `ab`, `sa`) are removed **only** when they occur as distinct word tokens at the **end** of the normalized business name.
- Suffixes appearing at the beginning or middle of names (e.g. `Limited Express Logistics`, `The Company Store`) are strictly preserved.
- If stripping all suffixes would result in an empty string, the original normalized name is preserved.

#### B. Country Normalization (Open-Set)
- Country is treated as an open-set field. Unseen or rare countries are **never** rejected, filtered, or dropped.
- Canonical alias mappings standardize common representations (e.g., `India`, `ind`, `INDIA` -> `in`; `U.S.A.`, `United States` -> `us`; `Deutschland` -> `de`).
- All other country strings pass through cleanly formatted in lowercase.

#### C. Memory Strategy & Streaming
- Processing is performed in streaming chunks (`chunksize=50000`).
- Chunks are normalized and streamed into temporary local storage before atomic transfer to destination (local directory or multi-part S3 upload).
- Full multi-gigabyte datasets are never buffered in RAM simultaneously.

### 4. Running Data Normalization

#### Local Mode (Sample Data)

```bash
python scripts/normalize_data.py \
    --input ./tests/fixtures/sample_data \
    --output ./artifacts/normalized/v001
```

#### S3 Mode (SageMaker Execution)

```bash
python scripts/normalize_data.py \
    --input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
    --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
    --aws-region ap-southeast-2 \
    --chunksize 50000
```

### 5. Reproducibility & Normalization Report

The normalization pipeline persists `normalization_report.json` containing:
- Exact artifact version (`v001`), timestamp, and CLI parameters
- Input and output URI locations
- File-level before/after row count assertions (must match exactly)
- Empty/null counts before vs. after normalization
- Complete list of recognized legal suffixes and country aliases
- Execution duration and integrity check statuses

---

## Experiment Tracking

All experimental runs, git hashes, pipeline versions, hyperparameters, and validation metrics are logged in [results.csv](file:///Users/arunraj/amazon-ml-challenge/amazon-ml-challenge-2026/experiments/results.csv).
