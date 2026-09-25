"""Data validation engine for Phase 1 of Amazon ML Challenge 2026.

Performs:
1. File existence validation for the 7 official TSV files.
2. Tab-separated schema validation for source and ground-truth files.
3. Data quality checks (null, empty string, whitespace-only counts).
4. Entity ID integrity and uniqueness checks.
5. Ground-truth reference integrity (Source 1 existence, match existence in Source 2 or 3).
6. Open-set country statistics generation.
7. Memory-conscious chunked processing.
8. Generation of machine-readable validation reports and dataset statistics.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from src.data.data_source import DataSource, LocalDataSource, create_data_source
from src.utils.logger import setup_logger

EXPECTED_FILES = [
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
]

SOURCE_EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_EXPECTED_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

MAX_DETAILED_ERRORS = 100


@dataclass
class ColumnQualityStats:
    """Statistics for a single column's quality."""
    null_count: int = 0
    empty_string_count: int = 0
    whitespace_only_count: int = 0
    valid_count: int = 0
    total_missing_or_blank: int = 0


@dataclass
class SourceFileStats:
    """Statistics for a single source TSV file."""
    file_path: str
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    column_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    unique_entity_id_count: int = 0
    duplicate_entity_id_count: int = 0
    missing_entity_id_count: int = 0
    country_distribution: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GroundTruthStats:
    """Statistics for train_ground_truth.tsv."""
    file_path: str = "train/train_ground_truth.tsv"
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    empty_match_count: int = 0
    non_empty_match_count: int = 0
    missing_source1_id_count: int = 0
    invalid_source1_reference_count: int = 0
    invalid_match_reference_count: int = 0
    duplicate_in_match_list_count: int = 0
    malformed_reference_count: int = 0


@dataclass
class ValidationResult:
    """Consolidated validation outcome and metrics."""
    status: str  # 'PASS' or 'FAIL'
    timestamp: str
    input_uri: str
    files_checked: int
    files_present: List[str]
    files_missing: List[str]
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataValidator:
    """Validates Amazon ML Challenge 2026 dataset files using chunked streaming."""

    def __init__(
        self,
        data_source: DataSource,
        chunksize: int = 50000,
        logger: Optional[logging.Logger] = None,
    ):
        self.data_source = data_source
        self.chunksize = chunksize
        self.logger = logger or setup_logger(name="data_validator")

    def _inspect_cell_quality(self, val: Any) -> str:
        """Categorizes a cell value into: 'null', 'empty', 'whitespace', or 'valid'."""
        if val is None or pd.isna(val) or str(val) in {"<NA>", "NaN", "null", "NULL", "None"}:
            return "null"
        s = str(val)
        if len(s) == 0:
            return "empty"
        if len(s.strip()) == 0:
            return "whitespace"
        return "valid"

    def validate_file_existence(self) -> Tuple[List[str], List[str], List[str]]:
        """Checks for all 7 required challenge files."""
        present: List[str] = []
        missing: List[str] = []
        errors: List[str] = []

        for rel_path in EXPECTED_FILES:
            if self.data_source.exists(rel_path):
                present.append(rel_path)
            else:
                missing.append(rel_path)
                errors.append(f"Required file missing: {rel_path}")

        return present, missing, errors

    def validate_source_file(
        self,
        rel_path: str,
        collect_entity_ids: bool = False,
    ) -> Tuple[SourceFileStats, Set[str], List[str], List[str]]:
        """Validates a source TSV file using chunked streaming.

        Returns:
            Tuple of (SourceFileStats, entity_id_set, errors, warnings).
        """
        stats = SourceFileStats(file_path=rel_path)
        errors: List[str] = []
        warnings: List[str] = []
        entity_ids: Set[str] = set()

        seen_ids: Set[str] = set()
        duplicate_id_count = 0
        missing_id_count = 0

        col_quality: Dict[str, ColumnQualityStats] = {
            col: ColumnQualityStats() for col in SOURCE_EXPECTED_COLUMNS
        }
        country_counter: Counter[str] = Counter()

        total_rows = 0
        header_checked = False

        try:
            chunk_iter = self.data_source.read_chunks(
                rel_path=rel_path,
                sep="\t",
                chunksize=self.chunksize,
            )

            for chunk_idx, chunk in enumerate(chunk_iter):
                if not header_checked:
                    stats.columns = list(chunk.columns)
                    missing_cols = [c for c in SOURCE_EXPECTED_COLUMNS if c not in chunk.columns]
                    if missing_cols:
                        errors.append(f"{rel_path}: Missing expected column(s): {missing_cols}")
                    extra_cols = [c for c in chunk.columns if c not in SOURCE_EXPECTED_COLUMNS]
                    if extra_cols:
                        warnings.append(f"{rel_path}: Unexpected additional column(s): {extra_cols}")
                    header_checked = True

                chunk_rows = len(chunk)
                total_rows += chunk_rows

                # Analyze columns present
                for col in SOURCE_EXPECTED_COLUMNS:
                    if col not in chunk.columns:
                        continue
                    q_stats = col_quality[col]
                    series = chunk[col]

                    for val in series:
                        cat = self._inspect_cell_quality(val)
                        if cat == "null":
                            q_stats.null_count += 1
                        elif cat == "empty":
                            q_stats.empty_string_count += 1
                        elif cat == "whitespace":
                            q_stats.whitespace_only_count += 1
                        else:
                            q_stats.valid_count += 1

                # Entity ID uniqueness and integrity
                if "entity_id" in chunk.columns:
                    for raw_id in chunk["entity_id"]:
                        cat = self._inspect_cell_quality(raw_id)
                        if cat != "valid":
                            missing_id_count += 1
                            if len(errors) < MAX_DETAILED_ERRORS:
                                errors.append(f"{rel_path}: Missing or blank entity_id encountered")
                        else:
                            eid = str(raw_id).strip()
                            if eid in seen_ids:
                                duplicate_id_count += 1
                                if len(warnings) < MAX_DETAILED_ERRORS:
                                    warnings.append(f"{rel_path}: Duplicate entity_id '{eid}' detected")
                            else:
                                seen_ids.add(eid)
                                if collect_entity_ids:
                                    entity_ids.add(eid)

                # Country distribution
                if "country" in chunk.columns:
                    for raw_country in chunk["country"]:
                        cat = self._inspect_cell_quality(raw_country)
                        if cat != "valid":
                            country_counter["<MISSING>"] += 1
                        else:
                            country_counter[str(raw_country).strip()] += 1

            if not header_checked:
                errors.append(f"{rel_path}: File is completely empty (no header)")

        except Exception as e:
            errors.append(f"{rel_path}: Failed reading or parsing TSV: {str(e)}")

        # Finalize stats
        stats.row_count = total_rows
        stats.unique_entity_id_count = len(seen_ids)
        stats.duplicate_entity_id_count = duplicate_id_count
        stats.missing_entity_id_count = missing_id_count

        if missing_id_count > 0:
            errors.append(f"{rel_path}: Total missing/blank entity IDs: {missing_id_count}")
        if duplicate_id_count > 0:
            warnings.append(f"{rel_path}: Total duplicate entity IDs: {duplicate_id_count}")

        # Column stats mapping
        for col, q in col_quality.items():
            q.total_missing_or_blank = q.null_count + q.empty_string_count + q.whitespace_only_count
            stats.column_stats[col] = asdict(q)

        # Country distribution sorted by frequency
        dist = []
        for country, count in country_counter.most_common():
            pct = round((count / total_rows * 100.0), 4) if total_rows > 0 else 0.0
            dist.append({"country": country, "row_count": count, "percentage": pct})
        stats.country_distribution = dist

        return stats, entity_ids, errors, warnings

    def validate_ground_truth(
        self,
        rel_path: str,
        train_s1_ids: Set[str],
        train_target_ids: Set[str],
    ) -> Tuple[GroundTruthStats, List[str], List[str]]:
        """Validates train_ground_truth.tsv against train sources.

        Args:
            rel_path: Path to ground truth file (train/train_ground_truth.tsv).
            train_s1_ids: Set of entity_ids from train_source1.tsv.
            train_target_ids: Set of entity_ids from train_source2 and train_source3.

        Returns:
            Tuple of (GroundTruthStats, errors, warnings).
        """
        stats = GroundTruthStats(file_path=rel_path)
        errors: List[str] = []
        warnings: List[str] = []

        total_rows = 0
        header_checked = False

        try:
            chunk_iter = self.data_source.read_chunks(
                rel_path=rel_path,
                sep="\t",
                chunksize=self.chunksize,
            )

            for chunk in chunk_iter:
                if not header_checked:
                    stats.columns = list(chunk.columns)
                    missing_cols = [c for c in GROUND_TRUTH_EXPECTED_COLUMNS if c not in chunk.columns]
                    if missing_cols:
                        errors.append(f"{rel_path}: Missing expected column(s): {missing_cols}")
                    header_checked = True

                total_rows += len(chunk)

                if "source1_entity_id" in chunk.columns and "matched_entity_ids" in chunk.columns:
                    for _, row in chunk.iterrows():
                        s1_val = row["source1_entity_id"]
                        matched_val = row["matched_entity_ids"]

                        s1_cat = self._inspect_cell_quality(s1_val)
                        if s1_cat != "valid":
                            stats.missing_source1_id_count += 1
                            if len(errors) < MAX_DETAILED_ERRORS:
                                errors.append(f"{rel_path}: Missing or blank source1_entity_id in ground truth")
                            s1_id = ""
                        else:
                            s1_id = str(s1_val).strip()
                            if train_s1_ids and s1_id not in train_s1_ids:
                                stats.invalid_source1_reference_count += 1
                                if len(errors) < MAX_DETAILED_ERRORS:
                                    errors.append(
                                        f"{rel_path}: source1_entity_id '{s1_id}' not found in train/train_source1.tsv"
                                    )

                        matched_cat = self._inspect_cell_quality(matched_val)
                        if matched_cat != "valid":
                            # 0 matches is completely valid in ground truth
                            stats.empty_match_count += 1
                        else:
                            stats.non_empty_match_count += 1
                            raw_str = str(matched_val)
                            tokens = raw_str.split(",")
                            seen_row_matches: Set[str] = set()

                            for token in tokens:
                                match_id = token.strip()
                                if not match_id:
                                    stats.malformed_reference_count += 1
                                    if len(errors) < MAX_DETAILED_ERRORS:
                                        errors.append(
                                            f"{rel_path}: Malformed empty token in matched_entity_ids '{raw_str}'"
                                        )
                                    continue

                                if match_id in seen_row_matches:
                                    stats.duplicate_in_match_list_count += 1
                                    if len(warnings) < MAX_DETAILED_ERRORS:
                                        warnings.append(
                                            f"{rel_path}: Duplicate match ID '{match_id}' for source1_id '{s1_id}'"
                                        )
                                else:
                                    seen_row_matches.add(match_id)

                                if train_target_ids and match_id not in train_target_ids:
                                    stats.invalid_match_reference_count += 1
                                    if len(errors) < MAX_DETAILED_ERRORS:
                                        errors.append(
                                            f"{rel_path}: Matched entity ID '{match_id}' not found in Source 2 or 3"
                                        )

            if not header_checked:
                errors.append(f"{rel_path}: File is completely empty (no header)")

        except Exception as e:
            errors.append(f"{rel_path}: Failed reading or parsing TSV: {str(e)}")

        stats.row_count = total_rows

        if stats.missing_source1_id_count > 0:
            errors.append(f"{rel_path}: Total missing source1 entity IDs: {stats.missing_source1_id_count}")
        if stats.invalid_source1_reference_count > 0:
            errors.append(
                f"{rel_path}: Total invalid source1 references: {stats.invalid_source1_reference_count}"
            )
        if stats.invalid_match_reference_count > 0:
            errors.append(
                f"{rel_path}: Total invalid matched entity references: {stats.invalid_match_reference_count}"
            )
        if stats.malformed_reference_count > 0:
            errors.append(
                f"{rel_path}: Total malformed matched references: {stats.malformed_reference_count}"
            )
        if stats.duplicate_in_match_list_count > 0:
            warnings.append(
                f"{rel_path}: Total duplicate matches in match lists: {stats.duplicate_in_match_list_count}"
            )

        return stats, errors, warnings

    def validate_all(self) -> ValidationResult:
        """Executes full validation across all 7 challenge files."""
        self.logger.info("Starting dataset validation on: %s", self.data_source.get_uri())

        all_errors: List[str] = []
        all_warnings: List[str] = []
        dataset_stats: Dict[str, Any] = {"files": {}, "ground_truth": {}}

        # 1. Existence Check
        present, missing, exist_errors = self.validate_file_existence()
        all_errors.extend(exist_errors)

        if missing:
            self.logger.error("Dataset validation aborted: %d required file(s) missing", len(missing))
            return ValidationResult(
                status="FAIL",
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_uri=self.data_source.get_uri(),
                files_checked=len(EXPECTED_FILES),
                files_present=present,
                files_missing=missing,
                errors=all_errors,
                warnings=all_warnings,
                statistics=dataset_stats,
            )

        # 2. Source Files Validation & ID caching for Ground Truth
        train_s1_ids: Set[str] = set()
        train_target_ids: Set[str] = set()

        source_files = [
            ("train/train_source1.tsv", True, "s1"),
            ("train/train_source2.tsv", True, "target"),
            ("train/train_source3.tsv", True, "target"),
            ("test/test_source1.tsv", False, None),
            ("test/test_source2.tsv", False, None),
            ("test/test_source3.tsv", False, None),
        ]

        total_source_rows = 0

        for file_path, collect_ids, role in source_files:
            self.logger.info("Validating source file: %s ...", file_path)
            s_stats, e_ids, s_errors, s_warnings = self.validate_source_file(
                rel_path=file_path,
                collect_entity_ids=collect_ids,
            )
            dataset_stats["files"][file_path] = asdict(s_stats)
            all_errors.extend(s_errors)
            all_warnings.extend(s_warnings)
            total_source_rows += s_stats.row_count

            if role == "s1":
                train_s1_ids = e_ids
            elif role == "target":
                train_target_ids.update(e_ids)

        # 3. Ground Truth Validation
        gt_path = "train/train_ground_truth.tsv"
        self.logger.info("Validating ground truth file: %s ...", gt_path)
        gt_stats, gt_errors, gt_warnings = self.validate_ground_truth(
            rel_path=gt_path,
            train_s1_ids=train_s1_ids,
            train_target_ids=train_target_ids,
        )
        dataset_stats["ground_truth"] = asdict(gt_stats)
        all_errors.extend(gt_errors)
        all_warnings.extend(gt_warnings)

        # Overall summary
        dataset_stats["summary"] = {
            "total_source_rows": total_source_rows,
            "ground_truth_rows": gt_stats.row_count,
            "ground_truth_empty_matches": gt_stats.empty_match_count,
            "ground_truth_non_empty_matches": gt_stats.non_empty_match_count,
            "total_errors": len(all_errors),
            "total_warnings": len(all_warnings),
        }

        status = "FAIL" if all_errors else "PASS"
        self.logger.info("Validation completed. Status: %s (Errors: %d, Warnings: %d)", status, len(all_errors), len(all_warnings))

        return ValidationResult(
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_uri=self.data_source.get_uri(),
            files_checked=len(EXPECTED_FILES),
            files_present=present,
            files_missing=missing,
            errors=all_errors,
            warnings=all_warnings,
            statistics=dataset_stats,
        )

    def write_reports(
        self,
        output_source_or_path: Union[DataSource, str, Path],
        result: ValidationResult,
    ) -> Tuple[str, str]:
        """Saves validation_report.json and dataset_stats.json to target output location.

        Args:
            output_source_or_path: Target DataSource, local directory, or s3:// URI.
            result: The ValidationResult to persist.

        Returns:
            Tuple of (validation_report_uri, dataset_stats_uri).
        """
        if isinstance(output_source_or_path, DataSource):
            out_ds = output_source_or_path
        else:
            out_ds = create_data_source(output_source_or_path)

        report_content = json.dumps(result.to_dict(), indent=2)
        stats_content = json.dumps(result.statistics, indent=2)

        rep_uri = out_ds.write_text("validation_report.json", report_content)
        stats_uri = out_ds.write_text("dataset_stats.json", stats_content)

        self.logger.info("Saved validation report: %s", rep_uri)
        self.logger.info("Saved dataset statistics: %s", stats_uri)
        return rep_uri, stats_uri
