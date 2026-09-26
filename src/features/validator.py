"""Validation and integrity verification for Phase 6 feature matrices."""

from dataclasses import asdict, dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from src.features.config import FEATURE_NAMES, FEATURE_VERSION

logger = logging.getLogger(__name__)


@dataclass
class FeatureValidationReport:
    """Encapsulates validation results for feature matrices."""

    status: str  # "PASS" or "FAIL"
    checks: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureValidator:
    """Validates streaming feature outputs for mathematical integrity and schema compliance."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger

    def validate_features_batch(
        self,
        features: List[List[float]],
        feature_names: List[str],
        is_candidate_dataset: bool = False,
        labels: Optional[List[int]] = None,
    ) -> Tuple[int, int, int]:
        """Validates a batch of extracted feature rows.

        Returns:
            Tuple of (valid_rows, nan_inf_count, invalid_dim_count).
        """
        expected_len = len(feature_names)
        nan_inf_count = 0
        invalid_dim_count = 0

        for row in features:
            if len(row) != expected_len:
                invalid_dim_count += 1
                continue
            for val in row:
                if math.isnan(val) or math.isinf(val):
                    nan_inf_count += 1
                    break

        valid_count = max(0, len(features) - invalid_dim_count - nan_inf_count)
        return valid_count, nan_inf_count, invalid_dim_count

    def build_report(
        self,
        dataset_name: str,
        total_rows_processed: int,
        feature_names: List[str],
        nan_inf_count: int,
        invalid_dim_count: int,
        labels_present: bool,
        is_candidate_dataset: bool = False,
        expected_rows: Optional[int] = None,
        git_commit: Optional[str] = None,
    ) -> FeatureValidationReport:
        """Constructs a comprehensive validation report for the feature dataset."""
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, str] = {}

        # Check 1: Feature dimension and names match schema
        if len(feature_names) != len(FEATURE_NAMES) or feature_names != FEATURE_NAMES:
            errors.append(
                f"Rule 1 Violation: Feature names do not match schema exactly ({len(feature_names)} vs {len(FEATURE_NAMES)})"
            )
            checks["1_feature_schema_match"] = "FAIL"
        else:
            checks["1_feature_schema_match"] = "PASS"

        # Check 2: Row dimension consistency
        if invalid_dim_count > 0:
            errors.append(f"Rule 2 Violation: {invalid_dim_count} rows have invalid feature dimensionality")
            checks["2_row_dimension_consistency"] = "FAIL"
        else:
            checks["2_row_dimension_consistency"] = "PASS"

        # Check 3: NaN and Inf detection
        if nan_inf_count > 0:
            errors.append(f"Rule 3 Violation: {nan_inf_count} rows contain NaN or infinite values")
            checks["3_no_nan_or_inf"] = "FAIL"
        else:
            checks["3_no_nan_or_inf"] = "PASS"

        # Check 4: No labels in candidate feature matrix
        if is_candidate_dataset and labels_present:
            errors.append("Rule 4 Violation: Candidate feature matrix contains target match labels (target leakage!)")
            checks["4_no_candidate_labels"] = "FAIL"
        else:
            checks["4_no_candidate_labels"] = "PASS"

        # Check 5: Row count consistency
        if expected_rows is not None and total_rows_processed != expected_rows:
            warnings.append(
                f"Rule 5 Warning: Processed rows ({total_rows_processed}) != expected ({expected_rows})"
            )
            checks["5_row_count_consistency"] = "WARN"
        else:
            checks["5_row_count_consistency"] = "PASS"

        # Check 6: Git version recorded
        if not git_commit or git_commit == "unknown":
            warnings.append("Rule 6 Warning: Git commit hash is unrecorded.")
            checks["6_git_commit"] = "WARN"
        else:
            checks["6_git_commit"] = "PASS"

        status = "FAIL" if errors else "PASS"

        metrics_summary = {
            "dataset_name": dataset_name,
            "feature_version": FEATURE_VERSION,
            "total_rows_processed": total_rows_processed,
            "feature_count": len(feature_names),
            "nan_or_inf_rows": nan_inf_count,
            "invalid_dim_rows": invalid_dim_count,
            "labels_present": labels_present,
        }

        self.logger.info(
            "Phase 6 Validation for %s complete: Status=%s (Errors=%d, Warnings=%d, Rows=%d)",
            dataset_name,
            status,
            len(errors),
            len(warnings),
            total_rows_processed,
        )

        return FeatureValidationReport(
            status=status,
            checks=checks,
            errors=errors,
            warnings=warnings,
            metrics_summary=metrics_summary,
        )
