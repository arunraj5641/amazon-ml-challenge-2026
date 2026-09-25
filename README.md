# Amazon ML Challenge 2026 - Entity Resolution

## Project Purpose

This repository houses the end-to-end entity-resolution pipeline for the **Amazon ML Challenge 2026**. The goal of the competition is to accurately resolve and link product entities across heterogeneous listings using scalable blocking, feature representations, and classification models.

This setup represents **Phase 0: Infrastructure and Project Setup**. It establishes a minimal, clean, reproducible foundation for configuration, testing, and logging before implementing any machine learning pipelines.

---

## Phase-Based Architecture

The architecture and workflow are strictly structured into distinct phases:

- **Phase 0**: Project infrastructure, directory layout, environment configuration, logging, testing, and smoke verification.
- **Phase 1+**: Exploratory data analysis, text/attribute normalization, candidate blocking, feature extraction, pair generation, model training, threshold calibration, and submission generation.

### System Architecture Flow

```
Mac + Antigravity (Local IDE & Prototyping)
        ↓
GitHub (Source of truth for all code, configs, and tests)
        ↓
Sage / SageMaker (Heavy computation, candidate generation, GPU training)
        ↓
S3 (Persistent storage for checkpoints, features, and model artifacts)
```

- **GitHub as Source of Truth**: All code, configurations, test suites, and documentation live in Git. Neither local machines nor remote servers maintain untracked code modifications.
- **Sage / SageMaker for Compute**: Prototyping and light testing happen locally, while resource-intensive steps (blocking, candidate generation, deep learning training) run on Sage/SageMaker.
- **S3 for Artifacts**: Large dataset splits, extracted embeddings, model checkpoints, and generated submission files are stored persistently in S3, never committed to version control.

---

## Directory Structure

```
.
├── .env.example              # Template for environment variables and secrets
├── .gitignore                # Exclusion rules for venv, data, artifacts, and secrets
├── README.md                 # Project documentation and getting started guide
├── requirements.txt          # Minimal setup and Phase 1 dependencies
├── configs/
│   └── default.yaml          # Project settings, paths, and environment defaults
├── data/
│   └── README.md             # Guidelines for local challenge dataset placement
├── artifacts/
│   └── .gitkeep              # Placeholder for model weights & generated artifacts
├── docs/
│   └── README.md             # Project documentation and phase tracking
├── experiments/
│   └── results.csv           # Experiment tracking log with required schema
├── scripts/
│   ├── normalize_data.py     # CLI runner for Phase 2 data normalization
│   ├── smoke_test.py         # Verification script for environment & Phase 1/2 pipelines
│   └── validate_data.py      # CLI runner for Phase 1 data ingestion & validation
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # YAML loader, path resolution, and env overrides
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_source.py    # LocalDataSource and S3DataSource abstractions
│   │   └── validator.py      # TSV schema, quality, entity ID, and GT validator
│   ├── normalization/
│   │   ├── __init__.py
│   │   ├── normalizer.py     # Deterministic text & open-set country normalization
│   │   └── pipeline.py       # Streaming chunked normalization pipeline
│   └── utils/
│       ├── __init__.py
│       └── logger.py         # Standardized logging setup
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   └── sample_data/      # Minimal synthetic TSV fixtures (7 files)
    ├── test_cli.py           # Tests for CLI exit codes and execution
    ├── test_config.py        # Tests for configuration and logging
    ├── test_data_source.py   # Tests for Local and S3 data sources
    ├── test_normalization.py # Tests for Phase 2 text normalization & pipeline
    ├── test_smoke.py         # Tests for smoke test runner and repo conventions
    └── test_validation.py    # Tests for schema, quality, ID, and GT validation
```

---

## Local Development Workflow

### 1. Activate the Virtual Environment

From the project root:

```bash
# On macOS / Linux:
source .venv/bin/activate
```

If the virtual environment does not yet exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Copy the `.env.example` template:

```bash
cp .env.example .env
```

Customize settings (e.g. `ENVIRONMENT`, `LOG_LEVEL`) as needed. Note that `.env` is gitignored and will never be committed.

### 3. Run the Smoke Test

Execute the smoke test to verify directories, configuration loading, results schema, and logger functionality:

```bash
python scripts/smoke_test.py
```

### 4. Run Phase 1 Data Ingestion & Validation

Validate local sample datasets:

```bash
python scripts/validate_data.py \
    --input ./tests/fixtures/sample_data \
    --output ./artifacts/validation
```

Or validate the official dataset in SageMaker against S3:

```bash
python scripts/validate_data.py \
    --input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
    --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/validation/v001/ \
    --aws-region ap-southeast-2
```

### 5. Run Phase 2 Data Normalization

Normalize local sample datasets:

```bash
python scripts/normalize_data.py \
    --input ./tests/fixtures/sample_data \
    --output ./artifacts/normalized/v001
```

Or normalize the official dataset in SageMaker against S3:

```bash
python scripts/normalize_data.py \
    --input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
    --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
    --aws-region ap-southeast-2
```

### 6. Run the Test Suite

Run unit and integration tests using `pytest`:

```bash
pytest
```

---

## Experiment Tracking

All experiments must be recorded in `experiments/results.csv` using the following schema:

```csv
experiment_id,git_commit,blocking_version,feature_version,model_version,threshold,precision,recall,f0_5,leaderboard_score,notes
```

Every experiment run must reference its exact git commit hash for full reproducibility.
