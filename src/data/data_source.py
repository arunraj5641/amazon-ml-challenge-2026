"""Data source abstraction supporting local filesystem and Amazon S3.

Provides a unified interface for file existence checks, chunked streaming
reading, and artifact writing without coupling downstream logic to storage backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union
from urllib.parse import urlparse

import pandas as pd


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parses an S3 URI (s3://bucket/prefix) into bucket and prefix components.

    Args:
        uri: S3 URI string.

    Returns:
        Tuple of (bucket_name, prefix_string). Prefix has leading and trailing slashes stripped.

    Raises:
        ValueError: If URI scheme is not 's3'.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI scheme '{parsed.scheme}': expected 's3://bucket/key'")

    bucket = parsed.netloc
    if not bucket:
        raise ValueError(f"S3 URI '{uri}' is missing a bucket name")

    prefix = parsed.path.strip("/")
    return bucket, prefix


class DataSource(ABC):
    """Abstract base class representing a data source (local or remote)."""

    @abstractmethod
    def exists(self, rel_path: str) -> bool:
        """Check if file exists at rel_path."""
        pass

    @abstractmethod
    def read_table(self, rel_path: str, sep: str = "\t", **kwargs) -> pd.DataFrame:
        """Read an entire tabular file into a pandas DataFrame."""
        pass

    @abstractmethod
    def read_chunks(
        self, rel_path: str, sep: str = "\t", chunksize: int = 50000, **kwargs
    ) -> Iterator[pd.DataFrame]:
        """Read a tabular file in chunks to minimize memory consumption."""
        pass

    @abstractmethod
    def write_text(self, rel_path: str, content: str) -> str:
        """Write text to target path, returning the destination URI or path."""
        pass

    @abstractmethod
    def get_uri(self, rel_path: str = "") -> str:
        """Returns the full URI or path representation."""
        pass


class LocalDataSource(DataSource):
    """Local filesystem implementation of DataSource."""

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir).resolve()

    def exists(self, rel_path: str) -> bool:
        target = self.base_dir / rel_path
        return target.is_file()

    def read_table(self, rel_path: str, sep: str = "\t", **kwargs) -> pd.DataFrame:
        target = self.base_dir / rel_path
        if not target.is_file():
            raise FileNotFoundError(f"Local file not found: {target}")
        default_kwargs = {"dtype": str, "keep_default_na": False}
        default_kwargs.update(kwargs)
        return pd.read_csv(target, sep=sep, **default_kwargs)

    def read_chunks(
        self, rel_path: str, sep: str = "\t", chunksize: int = 50000, **kwargs
    ) -> Iterator[pd.DataFrame]:
        target = self.base_dir / rel_path
        if not target.is_file():
            raise FileNotFoundError(f"Local file not found: {target}")
        default_kwargs = {"dtype": str, "keep_default_na": False}
        default_kwargs.update(kwargs)
        return pd.read_csv(target, sep=sep, chunksize=chunksize, **default_kwargs)

    def write_text(self, rel_path: str, content: str) -> str:
        target = self.base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)

    def get_uri(self, rel_path: str = "") -> str:
        if rel_path:
            return str(self.base_dir / rel_path)
        return str(self.base_dir)


class S3DataSource(DataSource):
    """Amazon S3 implementation of DataSource.

    Relies on standard ambient AWS credentials / IAM role in SageMaker.
    Accepts an optional s3_client for testing and dependency injection.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
        s3_client: Optional[Any] = None,
    ):
        self.bucket = bucket.strip()
        self.prefix = prefix.strip().strip("/")
        self.region = region
        self._client = s3_client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            kwargs: Dict[str, Any] = {}
            if self.region:
                kwargs["region_name"] = self.region
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _resolve_key(self, rel_path: str) -> str:
        clean_rel = rel_path.strip().lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{clean_rel}"
        return clean_rel

    def exists(self, rel_path: str) -> bool:
        import botocore.exceptions

        key = self._resolve_key(rel_path)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def read_table(self, rel_path: str, sep: str = "\t", **kwargs) -> pd.DataFrame:
        key = self._resolve_key(rel_path)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        default_kwargs = {"dtype": str, "keep_default_na": False}
        default_kwargs.update(kwargs)
        return pd.read_csv(response["Body"], sep=sep, **default_kwargs)

    def read_chunks(
        self, rel_path: str, sep: str = "\t", chunksize: int = 50000, **kwargs
    ) -> Iterator[pd.DataFrame]:
        key = self._resolve_key(rel_path)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        default_kwargs = {"dtype": str, "keep_default_na": False}
        default_kwargs.update(kwargs)
        return pd.read_csv(response["Body"], sep=sep, chunksize=chunksize, **default_kwargs)

    def write_text(self, rel_path: str, content: str) -> str:
        key = self._resolve_key(rel_path)
        content_type = "application/json" if rel_path.endswith(".json") else "text/plain"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType=content_type,
        )
        return f"s3://{self.bucket}/{key}"

    def get_uri(self, rel_path: str = "") -> str:
        key = self._resolve_key(rel_path) if rel_path else self.prefix
        if key:
            return f"s3://{self.bucket}/{key}"
        return f"s3://{self.bucket}"


def create_data_source(
    path_or_uri: Union[str, Path],
    region: Optional[str] = None,
    s3_client: Optional[Any] = None,
) -> DataSource:
    """Factory creating appropriate DataSource based on input path or URI.

    Args:
        path_or_uri: Local path string or S3 URI (s3://...).
        region: Optional AWS region for S3.
        s3_client: Optional boto3 client for testing/injection.

    Returns:
        Instance of LocalDataSource or S3DataSource.
    """
    path_str = str(path_or_uri).strip()
    if path_str.startswith("s3://"):
        bucket, prefix = parse_s3_uri(path_str)
        return S3DataSource(
            bucket=bucket,
            prefix=prefix,
            region=region,
            s3_client=s3_client,
        )
    return LocalDataSource(base_dir=path_str)
