"""Normalization pipeline for Amazon ML Challenge 2026 Phase 2.

Processes the 6 source TSVs in memory-conscious streaming chunks, producing
derived normalized TSVs while preserving all raw fields and entity IDs.
Generates comprehensive normalization reports with data integrity verification.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from src.data.data_source import DataSource, LocalDataSource, S3DataSource, create_data_source
from src.normalization.normalizer import (
    COUNTRY_ALIASES,
    RECOGNIZED_LEGAL_SUFFIXES,
    UNICODE_NORMALIZATION_FORM,
    extract_business_name_core,
    normalize_business_address,
    normalize_business_address_alnum,
    normalize_business_name,
    normalize_business_name_alnum,
    normalize_country,
)
from src.utils.logger import setup_logger

NORMALIZATION_VERSION = "v001"

SOURCE_FILES_TO_NORMALIZE = [
    ("train/train_source1.tsv", "train/train_source1_normalized.tsv"),
    ("train/train_source2.tsv", "train/train_source2_normalized.tsv"),
    ("train/train_source3.tsv", "train/train_source3_normalized.tsv"),
    ("test/test_source1.tsv", "test/test_source1_normalized.tsv"),
    ("test/test_source2.tsv", "test/test_source2_normalized.tsv"),
    ("test/test_source3.tsv", "test/test_source3_normalized.tsv"),
]

EXPECTED_INPUT_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

NORMALIZED_COLUMNS_ADDED = [
    "business_name_normalized",
    "business_name_core",
    "business_name_alnum",
    "business_address_normalized",
    "business_address_alnum",
    "country_normalized",
]

FINAL_COLUMN_ORDER = [
    "entity_id",
    "business_name",
    "business_name_normalized",
    "business_name_core",
    "business_name_alnum",
    "business_address",
    "business_address_normalized",
    "business_address_alnum",
    "country",
    "country_normalized",
]


@dataclass
class FileNormalizationSummary:
    input_file: str
    output_file: str
    row_count_before: int = 0
    row_count_after: int = 0
    empty_null_before: Dict[str, int] = field(default_factory=dict)
    empty_null_after: Dict[str, int] = field(default_factory=dict)
    country_distribution: List[Dict[str, Any]] = field(default_factory=list)
    field_length_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    integrity_verified: bool = False
    errors: List[str] = field(default_factory=list)


class NormalizationPipeline:
    """Orchestrates memory-efficient chunked normalization over source TSVs."""

    def __init__(
        self,
        input_source: DataSource,
        output_source: DataSource,
        chunksize: int = 50000,
        version: str = NORMALIZATION_VERSION,
        logger: Optional[logging.Logger] = None,
    ):
        self.input_source = input_source
        self.output_source = output_source
        self.chunksize = chunksize
        self.version = version
        self.logger = logger or setup_logger(name="normalization_pipeline")

    def _count_empty_or_null(self, series: pd.Series) -> int:
        """Counts values that are null, empty, or whitespace-only."""
        def is_empty(val: Any) -> bool:
            if val is None or pd.isna(val):
                return True
            s = str(val).strip()
            return len(s) == 0 or s.lower() in ("nan", "null", "none", "<na>")

        return int(series.apply(is_empty).sum())

    def _normalize_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Applies normalization functions to a single chunk and reorders columns."""
        df = chunk.copy()

        df["business_name_normalized"] = df["business_name"].apply(normalize_business_name)
        df["business_name_core"] = df["business_name_normalized"].apply(extract_business_name_core)
        df["business_name_alnum"] = df["business_name_normalized"].apply(normalize_business_name_alnum)

        df["business_address_normalized"] = df["business_address"].apply(normalize_business_address)
        df["business_address_alnum"] = df["business_address_normalized"].apply(
            normalize_business_address_alnum
        )

        df["country_normalized"] = df["country"].apply(normalize_country)

        # Enforce deterministic column layout
        return df[FINAL_COLUMN_ORDER]

    def normalize_file(
        self,
        input_rel_path: str,
        output_rel_path: str,
    ) -> FileNormalizationSummary:
        """Normalizes a single TSV file in streaming chunks."""
        summary = FileNormalizationSummary(
            input_file=input_rel_path,
            output_file=output_rel_path,
            empty_null_before={c: 0 for c in EXPECTED_INPUT_COLUMNS},
            empty_null_after={c: 0 for c in NORMALIZED_COLUMNS_ADDED},
        )

        if not self.input_source.exists(input_rel_path):
            summary.errors.append(f"Source file not found: {input_rel_path}")
            self.logger.error("Source file missing: %s", input_rel_path)
            return summary

        self.logger.info("Normalizing %s -> %s (chunksize=%d)", input_rel_path, output_rel_path, self.chunksize)

        country_counter: Counter[str] = Counter()
        entity_ids_seen: Set[str] = set()
        lengths: Dict[str, List[int]] = {col: [] for col in NORMALIZED_COLUMNS_ADDED}

        total_rows_before = 0
        total_rows_after = 0

        # We stream into a local scratch file to ensure memory efficiency
        # and atomic upload to S3 if using S3DataSource
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            chunk_iter = self.input_source.read_chunks(
                rel_path=input_rel_path,
                sep="\t",
                chunksize=self.chunksize,
            )

            is_first_chunk = True

            for chunk_idx, chunk in enumerate(chunk_iter):
                rows_in_chunk = len(chunk)
                total_rows_before += rows_in_chunk

                # Verify expected columns present
                missing_cols = [c for c in EXPECTED_INPUT_COLUMNS if c not in chunk.columns]
                if missing_cols:
                    err = f"{input_rel_path}: Chunk {chunk_idx} missing expected columns: {missing_cols}"
                    summary.errors.append(err)
                    raise ValueError(err)

                # Track null counts before
                for col in EXPECTED_INPUT_COLUMNS:
                    summary.empty_null_before[col] += self._count_empty_or_null(chunk[col])

                # Normalize chunk
                norm_chunk = self._normalize_chunk(chunk)
                total_rows_after += len(norm_chunk)

                # Integrity: check entity IDs
                for eid in norm_chunk["entity_id"]:
                    entity_ids_seen.add(str(eid))

                # Track null counts after
                for col in NORMALIZED_COLUMNS_ADDED:
                    summary.empty_null_after[col] += self._count_empty_or_null(norm_chunk[col])
                    # Sample character lengths for statistics (capped sample to prevent memory growth)
                    if len(lengths[col]) < 10000:
                        lengths[col].extend(norm_chunk[col].str.len().tolist())

                # Track country distribution
                for c_val in norm_chunk["country_normalized"]:
                    country_counter[c_val if c_val else "<MISSING>"] += 1

                # Append to temp TSV file
                norm_chunk.to_csv(
                    temp_path,
                    sep="\t",
                    index=False,
                    mode="w" if is_first_chunk else "a",
                    header=is_first_chunk,
                )
                is_first_chunk = False

            # Transfer to destination
            if isinstance(self.output_source, LocalDataSource):
                target_dest = self.output_source.base_dir / output_rel_path
                target_dest.parent.mkdir(parents=True, exist_ok=True)
                temp_path.replace(target_dest)
            elif isinstance(self.output_source, S3DataSource):
                s3_key = self.output_source._resolve_key(output_rel_path)
                self.logger.info("Uploading normalized TSV to s3://%s/%s ...", self.output_source.bucket, s3_key)
                self.output_source.client.upload_file(str(temp_path), self.output_source.bucket, s3_key)
                if temp_path.exists():
                    temp_path.unlink()
            else:
                # Generic fallback via write_text
                content = temp_path.read_text(encoding="utf-8")
                self.output_source.write_text(output_rel_path, content)
                if temp_path.exists():
                    temp_path.unlink()

            summary.row_count_before = total_rows_before
            summary.row_count_after = total_rows_after

            # Data Integrity Assertions
            if total_rows_before != total_rows_after:
                summary.errors.append(
                    f"Row count mismatch: before={total_rows_before}, after={total_rows_after}"
                )

            if len(entity_ids_seen) != total_rows_before:
                # Warning or notice if duplicates were already present in raw data
                self.logger.warning(
                    "%s: unique entity IDs (%d) != total rows (%d)",
                    input_rel_path,
                    len(entity_ids_seen),
                    total_rows_before,
                )

            # Field length statistics
            for col, l_list in lengths.items():
                if l_list:
                    summary.field_length_stats[col] = {
                        "min_length": float(min(l_list)),
                        "max_length": float(max(l_list)),
                        "mean_length": round(float(sum(l_list) / len(l_list)), 2),
                    }

            # Country distribution list
            dist = []
            for c_name, count in country_counter.most_common():
                pct = round((count / total_rows_after * 100.0), 4) if total_rows_after > 0 else 0.0
                dist.append({"country": c_name, "count": count, "percentage": pct})
            summary.country_distribution = dist

            if not summary.errors:
                summary.integrity_verified = True

        except Exception as e:
            summary.errors.append(f"Exception during normalization: {str(e)}")
            self.logger.error("Error normalizing %s: %s", input_rel_path, e, exc_info=True)
            if temp_path.exists():
                temp_path.unlink()

        return summary

    def run(self) -> Dict[str, Any]:
        """Runs normalization across all 6 source TSVs and writes normalization_report.json."""
        start_time = time.time()
        self.logger.info("Starting Phase 2 Normalization Pipeline (version: %s)", self.version)

        file_summaries: Dict[str, Any] = {}
        all_errors: List[str] = []

        total_rows_processed = 0

        for input_rel, output_rel in SOURCE_FILES_TO_NORMALIZE:
            summary = self.normalize_file(input_rel, output_rel)
            file_summaries[input_rel] = asdict(summary)
            all_errors.extend(summary.errors)
            total_rows_processed += summary.row_count_after

        duration = round(time.time() - start_time, 2)
        status = "PASS" if not all_errors else "FAIL"

        report = {
            "status": status,
            "version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_location": self.input_source.get_uri(),
            "output_location": self.output_source.get_uri(),
            "processing_duration_seconds": duration,
            "files_processed": [out for _, out in SOURCE_FILES_TO_NORMALIZE],
            "total_rows_processed": total_rows_processed,
            "columns_added": NORMALIZED_COLUMNS_ADDED,
            "column_layout": FINAL_COLUMN_ORDER,
            "normalization_configuration": {
                "unicode_form": UNICODE_NORMALIZATION_FORM,
                "recognized_legal_suffixes_count": len(RECOGNIZED_LEGAL_SUFFIXES),
                "recognized_legal_suffixes": RECOGNIZED_LEGAL_SUFFIXES,
                "recognized_country_aliases_count": len(COUNTRY_ALIASES),
                "open_set_country_preservation": True,
            },
            "file_summaries": file_summaries,
            "errors": all_errors,
        }

        # Write report
        report_json = json.dumps(report, indent=2)
        report_uri = self.output_source.write_text("normalization_report.json", report_json)
        self.logger.info("Saved normalization report: %s", report_uri)

        self.logger.info(
            "Phase 2 normalization completed in %ss with status %s (Total rows: %d, Errors: %d)",
            duration,
            status,
            total_rows_processed,
            len(all_errors),
        )

        return report
