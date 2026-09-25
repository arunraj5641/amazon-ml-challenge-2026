"""Data ingestion and validation package for Phase 1."""

from src.data.data_source import (
    DataSource,
    LocalDataSource,
    S3DataSource,
    create_data_source,
    parse_s3_uri,
)
from src.data.validator import (
    DataValidator,
    ValidationResult,
    EXPECTED_FILES,
    SOURCE_EXPECTED_COLUMNS,
    GROUND_TRUTH_EXPECTED_COLUMNS,
)

__all__ = [
    "DataSource",
    "LocalDataSource",
    "S3DataSource",
    "create_data_source",
    "parse_s3_uri",
    "DataValidator",
    "ValidationResult",
    "EXPECTED_FILES",
    "SOURCE_EXPECTED_COLUMNS",
    "GROUND_TRUTH_EXPECTED_COLUMNS",
]
