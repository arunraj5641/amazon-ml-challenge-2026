"""Unit tests for DataSource abstraction (Local and S3)."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
import botocore.exceptions
import pytest

from src.data.data_source import (
    DataSource,
    LocalDataSource,
    S3DataSource,
    create_data_source,
    parse_s3_uri,
)


def test_parse_s3_uri_valid():
    bucket, prefix = parse_s3_uri("s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/")
    assert bucket == "sagemaker-ap-southeast-2-904290466033"
    assert prefix == "raw/dataset"

    b2, p2 = parse_s3_uri("s3://my-bucket/artifacts/validation/v001")
    assert b2 == "my-bucket"
    assert p2 == "artifacts/validation/v001"

    b3, p3 = parse_s3_uri("s3://root-bucket")
    assert b3 == "root-bucket"
    assert p3 == ""


def test_parse_s3_uri_invalid():
    with pytest.raises(ValueError, match="Invalid S3 URI scheme"):
        parse_s3_uri("https://s3.amazonaws.com/my-bucket/data")

    with pytest.raises(ValueError, match="missing a bucket name"):
        parse_s3_uri("s3://")


def test_local_data_source(tmp_path: Path):
    # Setup test file
    test_file = tmp_path / "train" / "test.tsv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("col_a\tcol_b\n1\tval1\n2\tval2\n", encoding="utf-8")

    ds = LocalDataSource(base_dir=tmp_path)
    assert ds.exists("train/test.tsv") is True
    assert ds.exists("train/nonexistent.tsv") is False

    # Read table
    df = ds.read_table("train/test.tsv")
    assert len(df) == 2
    assert list(df.columns) == ["col_a", "col_b"]

    # Read chunks
    chunks = list(ds.read_chunks("train/test.tsv", chunksize=1))
    assert len(chunks) == 2
    assert chunks[0].iloc[0]["col_a"] == "1"

    # Write text
    written_path = ds.write_text("output/report.txt", "sample report")
    assert Path(written_path).is_file()
    assert Path(written_path).read_text(encoding="utf-8") == "sample report"

    # Missing file error
    with pytest.raises(FileNotFoundError):
        ds.read_table("train/nonexistent.tsv")


def test_s3_data_source_mocked():
    """Verify S3DataSource works without real AWS credentials via mocked boto3 client."""
    mock_client = MagicMock()

    # Mock head_object for exists()
    def head_side_effect(Bucket, Key):
        if Key == "raw/dataset/train/train_source1.tsv":
            return {"ContentLength": 100}
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        raise botocore.exceptions.ClientError(error_response, "HeadObject")

    mock_client.head_object.side_effect = head_side_effect

    tsv_content = b"entity_id\tbusiness_name\tbusiness_address\tcountry\ne1\tAcme\t100 St\tUS\n"

    def get_side_effect(Bucket, Key):
        return {"Body": BytesIO(tsv_content)}

    mock_client.get_object.side_effect = get_side_effect

    s3_ds = S3DataSource(
        bucket="sagemaker-ap-southeast-2-904290466033",
        prefix="raw/dataset",
        region="ap-southeast-2",
        s3_client=mock_client,
    )

    assert s3_ds.exists("train/train_source1.tsv") is True
    assert s3_ds.exists("train/missing.tsv") is False

    df = s3_ds.read_table("train/train_source1.tsv")
    assert len(df) == 1
    assert df.iloc[0]["entity_id"] == "e1"

    # Mock writing
    uri = s3_ds.write_text("artifacts/report.json", '{"status": "PASS"}')
    assert uri == "s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/artifacts/report.json"
    mock_client.put_object.assert_called_once()


def test_create_data_source_factory():
    local_ds = create_data_source("./tests/fixtures/sample_data")
    assert isinstance(local_ds, LocalDataSource)

    s3_ds = create_data_source(
        "s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/",
        region="ap-southeast-2",
        s3_client=MagicMock(),
    )
    assert isinstance(s3_ds, S3DataSource)
    assert s3_ds.bucket == "sagemaker-ap-southeast-2-904290466033"
    assert s3_ds.prefix == "raw/dataset"
