"""Unit tests for Phase 1 data validation engine."""

import shutil
from pathlib import Path
import pytest

from src.data.data_source import LocalDataSource
from src.data.validator import (
    DataValidator,
    EXPECTED_FILES,
    SOURCE_EXPECTED_COLUMNS,
    GROUND_TRUTH_EXPECTED_COLUMNS,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_data"


def test_valid_sample_dataset_passes():
    """Verify that the clean sample dataset passes all validations with 0 errors."""
    ds = LocalDataSource(base_dir=FIXTURES_DIR)
    validator = DataValidator(data_source=ds, chunksize=10)
    result = validator.validate_all()

    assert result.status == "PASS"
    assert len(result.errors) == 0
    assert result.files_checked == 7
    assert len(result.files_present) == 7
    assert len(result.files_missing) == 0

    # Ground truth assertions
    gt_stats = result.statistics["ground_truth"]
    assert gt_stats["row_count"] == 4
    assert gt_stats["empty_match_count"] == 2
    assert gt_stats["non_empty_match_count"] == 2
    assert gt_stats["missing_source1_id_count"] == 0
    assert gt_stats["invalid_source1_reference_count"] == 0
    assert gt_stats["invalid_match_reference_count"] == 0


def test_missing_required_file_fails(tmp_path: Path):
    """Verify that omitting any of the 7 official files causes validation to fail."""
    # Copy sample data except test_source3.tsv
    shutil.copytree(FIXTURES_DIR, tmp_path / "data")
    (tmp_path / "data" / "test" / "test_source3.tsv").unlink()

    ds = LocalDataSource(base_dir=tmp_path / "data")
    validator = DataValidator(data_source=ds)
    result = validator.validate_all()

    assert result.status == "FAIL"
    assert "test/test_source3.tsv" in result.files_missing
    assert any("test/test_source3.tsv" in err for err in result.errors)


def test_schema_missing_column_fails(tmp_path: Path):
    """Verify missing required columns trigger schema validation errors."""
    shutil.copytree(FIXTURES_DIR, tmp_path / "data")
    # Replace train_source1.tsv with a file missing 'country'
    bad_source1 = (
        "entity_id\tbusiness_name\tbusiness_address\n"
        "s1_1\tAcme\t123 St\n"
    )
    (tmp_path / "data" / "train" / "train_source1.tsv").write_text(bad_source1, encoding="utf-8")

    ds = LocalDataSource(base_dir=tmp_path / "data")
    validator = DataValidator(data_source=ds)
    result = validator.validate_all()

    assert result.status == "FAIL"
    assert any("Missing expected column" in err for err in result.errors)


def test_data_quality_null_and_empty_detection(tmp_path: Path):
    """Verify nulls, empty strings, and whitespace values are categorized and counted."""
    shutil.copytree(FIXTURES_DIR, tmp_path / "data")
    # Add rows with empty string, whitespace only, and explicit null
    custom_s1 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "s1_1\t\t100 Main St\tUS\n"              # empty business_name
        "s1_2\tAcme Inc\t   \tUS\n"               # whitespace business_address
        "s1_3\tAcme Corp\t200 Elm St\tNaN\n"      # null country
    )
    (tmp_path / "data" / "train" / "train_source1.tsv").write_text(custom_s1, encoding="utf-8")

    ds = LocalDataSource(base_dir=tmp_path / "data")
    validator = DataValidator(data_source=ds)
    stats, _, errors, _ = validator.validate_source_file("train/train_source1.tsv")

    assert stats.row_count == 3
    # business_name empty string
    assert stats.column_stats["business_name"]["empty_string_count"] == 1
    # business_address whitespace
    assert stats.column_stats["business_address"]["whitespace_only_count"] == 1
    # country null
    assert stats.column_stats["country"]["null_count"] == 1


def test_entity_id_duplicate_and_missing(tmp_path: Path):
    """Verify duplicate and missing entity IDs are detected."""
    shutil.copytree(FIXTURES_DIR, tmp_path / "data")
    custom_s1 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "s1_1\tBiz 1\tAddr 1\tUS\n"
        "s1_1\tBiz 1 Dupe\tAddr 2\tUS\n"   # duplicate ID
        "\tBiz 3\tAddr 3\tUS\n"             # missing ID
    )
    (tmp_path / "data" / "train" / "train_source1.tsv").write_text(custom_s1, encoding="utf-8")

    ds = LocalDataSource(base_dir=tmp_path / "data")
    validator = DataValidator(data_source=ds)
    stats, _, errors, warnings = validator.validate_source_file("train/train_source1.tsv")

    assert stats.unique_entity_id_count == 1
    assert stats.duplicate_entity_id_count == 1
    assert stats.missing_entity_id_count == 1
    assert any("Duplicate entity_id" in w for w in warnings)
    assert any("Missing or blank entity_id" in err for err in errors)


def test_ground_truth_invalid_references(tmp_path: Path):
    """Verify invalid Source 1 or match references in ground truth fail validation."""
    shutil.copytree(FIXTURES_DIR, tmp_path / "data")

    # s1_999 does not exist in train_source1; s2_999 does not exist in train_source2/3
    bad_gt = (
        "source1_entity_id\tmatched_entity_ids\n"
        "s1_999\ts2_101\n"               # invalid source 1 reference
        "s1_101\ts2_999,s3_999\n"         # invalid matched references
        "\ts2_102\n"                      # missing source 1 ID
        "s1_102\ts2_101,,s3_101\n"        # malformed match list
    )
    (tmp_path / "data" / "train" / "train_ground_truth.tsv").write_text(bad_gt, encoding="utf-8")

    ds = LocalDataSource(base_dir=tmp_path / "data")
    validator = DataValidator(data_source=ds)
    result = validator.validate_all()

    assert result.status == "FAIL"
    gt_stats = result.statistics["ground_truth"]
    assert gt_stats["invalid_source1_reference_count"] == 1
    assert gt_stats["invalid_match_reference_count"] == 2
    assert gt_stats["missing_source1_id_count"] == 1
    assert gt_stats["malformed_reference_count"] == 1


def test_country_distribution_open_set():
    """Verify country statistics support arbitrary countries without hardcoding."""
    ds = LocalDataSource(base_dir=FIXTURES_DIR)
    validator = DataValidator(data_source=ds)
    stats, _, _, _ = validator.validate_source_file("train/train_source1.tsv")

    countries = {item["country"] for item in stats.country_distribution}
    # France and Japan are present in sample data
    assert "France" in countries
    assert "Japan" in countries
    assert "India" in countries
    assert "US" in countries
