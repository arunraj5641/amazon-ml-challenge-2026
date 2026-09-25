"""Unit tests for Phase 2 text normalization functions and pipeline."""

from pathlib import Path
import pytest

from src.data.data_source import LocalDataSource
from src.normalization.normalizer import (
    extract_business_name_core,
    normalize_business_address,
    normalize_business_address_alnum,
    normalize_business_name,
    normalize_business_name_alnum,
    normalize_country,
)
from src.normalization.pipeline import (
    FINAL_COLUMN_ORDER,
    NORMALIZED_COLUMNS_ADDED,
    NormalizationPipeline,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_data"


# ==============================================================================
# Business Name Normalization Tests
# ==============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("ABC Pvt. Ltd.", "abc pvt ltd"),
        ("ABC PRIVATE LIMITED", "abc private limited"),
        ("abc  pvt   ltd", "abc pvt ltd"),
        ("ABC, Pvt Ltd.", "abc pvt ltd"),
        ("AT&T Inc.", "at and t inc"),
        ("A+B Solutions", "a and b solutions"),
        ("Hewlett-Packard", "hewlett-packard"),
        ("McDonald's Stores", "mcdonald's stores"),
        ("U.S.A. Tech Ltd.", "usa tech ltd"),
        ("  Café   Express  ", "café express"),  # Unicode
        ("㈱ 日本会社", "株 日本会社"),  # Unicode NFKC ligature normalization
    ],
)
def test_normalize_business_name(raw, expected):
    result = normalize_business_name(raw)
    assert result == expected
    # Idempotence: f(f(x)) == f(x)
    assert normalize_business_name(result) == result


@pytest.mark.parametrize(
    "normalized,expected_core",
    [
        ("abc pvt ltd", "abc"),
        ("abc private limited", "abc"),
        ("abc inc", "abc"),
        ("abc corp", "abc"),
        ("abc co ltd", "abc"),
        ("berlin tech gmbh", "berlin tech"),
        ("nordic solutions ab", "nordic solutions"),
        ("limited express logistics", "limited express logistics"),  # Suffix at start preserved!
        ("the company store", "the company store"),  # Suffix in middle preserved!
        ("company ltd", "company"),  # Does not reduce to empty if multiple tokens
        ("ltd", "ltd"),  # Does not reduce standalone suffix to empty string
    ],
)
def test_extract_business_name_core(normalized, expected_core):
    core = extract_business_name_core(normalized)
    assert core == expected_core
    # Idempotence
    assert extract_business_name_core(core) == core


def test_normalize_business_name_alnum():
    assert normalize_business_name_alnum("Hewlett-Packard, Inc.") == "hewlett packard inc"
    assert normalize_business_name_alnum("AT&T Corp.") == "at and t corp"
    assert normalize_business_name_alnum("") == ""


# ==============================================================================
# Business Address Normalization Tests
# ==============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("123 Main St.", "123 main st"),
        ("123 Main Street", "123 main street"),
        ("10-A MG Road", "10-a mg road"),
        ("10 A MG Road", "10 a mg road"),
        ("12, Main Street, Chennai", "12 main street chennai"),
        ("Suite # 400; 5th Floor", "suite 400 5th floor"),
        ("45, Rue de Rivoli, 75001 Paris", "45 rue de rivoli 75001 paris"),
        ("1-1 Chiyoda-ku, Tokyo", "1-1 chiyoda-ku tokyo"),
    ],
)
def test_normalize_business_address(raw, expected):
    result = normalize_business_address(raw)
    assert result == expected
    # Idempotence: f(f(x)) == f(x)
    assert normalize_business_address(result) == result


def test_normalize_business_address_alnum():
    assert normalize_business_address_alnum("10-A MG Road, Bangalore") == "10 a mg road bangalore"
    assert normalize_business_address_alnum("1-1 Chiyoda-ku, Tokyo") == "1 1 chiyoda ku tokyo"
    assert normalize_business_address_alnum(None) == ""


# ==============================================================================
# Multilingual Unicode Preservation Tests
# ==============================================================================


@pytest.mark.parametrize(
    "raw_input,expected_name,expected_core,expected_alnum,expected_addr,expected_addr_alnum",
    [
        ("Café", "café", "café", "café", "café", "café"),
        ("São Paulo", "são paulo", "são paulo", "são paulo", "são paulo", "são paulo"),
        ("München", "münchen", "münchen", "münchen", "münchen", "münchen"),
        ("Zürich", "zürich", "zürich", "zürich", "zürich", "zürich"),
        ("東京", "東京", "東京", "東京", "東京", "東京"),
        ("北京", "北京", "北京", "北京", "北京", "北京"),
        ("Москва", "москва", "москва", "москва", "москва", "москва"),
        ("İstanbul", "i̇stanbul", "i̇stanbul", "i̇stanbul", "i̇stanbul", "i̇stanbul"),
    ],
)
def test_unicode_preservation_multilingual(
    raw_input,
    expected_name,
    expected_core,
    expected_alnum,
    expected_addr,
    expected_addr_alnum,
):
    """Verify that normalization preserves Unicode letters and scripts without stripping."""
    name_norm = normalize_business_name(raw_input)
    assert name_norm == expected_name
    assert normalize_business_name(name_norm) == name_norm  # Idempotence

    core = extract_business_name_core(name_norm)
    assert core == expected_core
    assert extract_business_name_core(core) == core  # Idempotence

    name_alnum = normalize_business_name_alnum(raw_input)
    assert name_alnum == expected_alnum
    assert normalize_business_name_alnum(name_alnum) == name_alnum  # Idempotence

    addr_norm = normalize_business_address(raw_input)
    assert addr_norm == expected_addr
    assert normalize_business_address(addr_norm) == addr_norm  # Idempotence

    addr_alnum = normalize_business_address_alnum(raw_input)
    assert addr_alnum == expected_addr_alnum
    assert normalize_business_address_alnum(addr_alnum) == addr_alnum  # Idempotence


# ==============================================================================
# Country Normalization Tests (Open-Set)
# ==============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        # Aliases mapped to 2-letter codes
        ("US", "us"),
        ("U.S.A.", "us"),
        ("United States", "us"),
        ("India", "in"),
        ("india", "in"),
        ("  INDIA  ", "in"),
        ("France", "fr"),
        ("Germany", "de"),
        ("Japan", "jp"),
        ("Australia", "au"),
        ("Canada", "ca"),
        ("Sweden", "se"),
        # Open-set unmapped countries preserved cleanly
        ("Brazil", "br"),
        ("New Zealand", "new zealand"),
        ("Kenya", "kenya"),
        ("South Africa", "south africa"),
    ],
)
def test_normalize_country_open_set(raw, expected):
    result = normalize_country(raw)
    assert result == expected
    # Idempotence: f(f(x)) == f(x)
    assert normalize_country(result) == result


# ==============================================================================
# Pipeline & Data Integrity Tests
# ==============================================================================


def test_normalization_pipeline_integrity(tmp_path: Path):
    """Verifies that NormalizationPipeline preserves row counts and entity IDs."""
    input_ds = LocalDataSource(base_dir=FIXTURES_DIR)
    output_ds = LocalDataSource(base_dir=tmp_path / "normalized")

    pipeline = NormalizationPipeline(
        input_source=input_ds,
        output_source=output_ds,
        chunksize=2,  # test chunking across small files
    )
    report = pipeline.run()

    assert report["status"] == "PASS"
    assert len(report["errors"]) == 0
    assert report["total_rows_processed"] == 15
    assert len(report["files_processed"]) == 6

    # Verify each output file
    for input_file, output_file in [
        ("train/train_source1.tsv", "train/train_source1_normalized.tsv"),
        ("train/train_source2.tsv", "train/train_source2_normalized.tsv"),
        ("train/train_source3.tsv", "train/train_source3_normalized.tsv"),
        ("test/test_source1.tsv", "test/test_source1_normalized.tsv"),
        ("test/test_source2.tsv", "test/test_source2_normalized.tsv"),
        ("test/test_source3.tsv", "test/test_source3_normalized.tsv"),
    ]:
        raw_df = input_ds.read_table(input_file)
        norm_df = output_ds.read_table(output_file)

        # 1. Row count preservation
        assert len(norm_df) == len(raw_df)

        # 2. Entity IDs preserved in order
        assert list(norm_df["entity_id"]) == list(raw_df["entity_id"])

        # 3. Raw fields strictly preserved
        assert list(norm_df["business_name"]) == list(raw_df["business_name"])
        assert list(norm_df["business_address"]) == list(raw_df["business_address"])
        assert list(norm_df["country"]) == list(raw_df["country"])

        # 4. Column layout
        assert list(norm_df.columns) == FINAL_COLUMN_ORDER

        # 5. Normalized fields are non-empty when raw fields are non-empty
        assert (norm_df["business_name_normalized"] != "").all()
        assert (norm_df["business_address_normalized"] != "").all()
        assert (norm_df["country_normalized"] != "").all()

    # Verify report was generated
    assert output_ds.exists("normalization_report.json")
