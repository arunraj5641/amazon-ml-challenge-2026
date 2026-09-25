# Challenge Data Directory

This directory stores datasets for the Amazon ML Challenge 2026 entity-resolution task.

## Rules & Guidelines

1. **Do Not Commit Datasets**: All dataset files (raw CSVs, Parquet files, extracted images/text) are strictly excluded from version control via `.gitignore`.
2. **Local Development**: Place raw and intermediate processed splits here when running pipeline steps locally.
3. **Remote Execution (Sage/SageMaker)**: Large datasets are persisted in S3 and synced into compute instances dynamically or streamed directly.
4. **Data Privacy**: Ensure no proprietary competition datasets or credentials leak into code, commits, or pull requests.
