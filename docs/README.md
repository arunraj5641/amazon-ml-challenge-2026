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

- **Phase 0: Infrastructure & Project Setup (Current)**
  - Clean project layout, configuration, logging, testing harness, and smoke test.
  - No ML logic, no dataset dependencies.
- **Future Phases (1+)**:
  - Exploratory data analysis & baseline normalization.
  - Blocking & Candidate generation.
  - Feature extraction & training-pair generation.
  - Model training, threshold calibration, and F0.5 evaluation.
  - Inference pipelines & submission generation.

## Experiment Tracking

All experimental runs, git hashes, pipeline versions, hyperparameters, and validation metrics are logged in [results.csv](file:///Users/arunraj/amazon-ml-challenge/amazon-ml-challenge-2026/experiments/results.csv).
