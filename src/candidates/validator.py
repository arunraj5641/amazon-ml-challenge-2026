"""Validation rules and report structure for Phase 5 Candidate Universe."""

from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ID Regexes supporting both underscore and hyphen
S1_ID_PATTERN = re.compile(r"^s1[_-].+", re.IGNORECASE)
S2_ID_PATTERN = re.compile(r"^s2[_-].+", re.IGNORECASE)
S3_ID_PATTERN = re.compile(r"^s3[_-].+", re.IGNORECASE)
TARGET_ID_PATTERN = re.compile(r"^(s2|s3)[_-].+", re.IGNORECASE)


@dataclass
class CandidateValidationReport:
    """Encapsulates validation results for Phase 5 candidate finalization."""

    status: str  # "PASS" or "FAIL"
    checks: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts report to dictionary representation."""
        return asdict(self)


class CandidateValidator:
    """Validates streaming candidate pairs for Phase 5 finalization."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger

    def build_report(
        self,
        total_input_count: int,
        total_output_count: int,
        duplicate_count: int,
        invalid_s1_count: int,
        invalid_target_count: int,
        inconsistent_source_count: int,
        test_leakage_count: int,
        ordering_violations: int,
        expected_candidate_count: Optional[int] = None,
        git_commit: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> CandidateValidationReport:
        """Builds a formal Phase 5 CandidateValidationReport from streaming counters."""
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, str] = {}

        # Check 1: S1 ID format
        if invalid_s1_count > 0:
            errors.append(f"Rule 1 Violation: {invalid_s1_count} invalid Source 1 entity IDs (must match s1_... or s1-...)")
            checks["1_s1_id_format"] = "FAIL"
        else:
            checks["1_s1_id_format"] = "PASS"

        # Check 2: Target ID format
        if invalid_target_count > 0:
            errors.append(f"Rule 2 Violation: {invalid_target_count} invalid Target entity IDs (must match s2_... or s3_...)")
            checks["2_target_id_format"] = "FAIL"
        else:
            checks["2_target_id_format"] = "PASS"

        # Check 3: Target source consistency
        if inconsistent_source_count > 0:
            errors.append(f"Rule 3 Violation: {inconsistent_source_count} rows with inconsistent target_source label")
            checks["3_target_source_consistency"] = "FAIL"
        else:
            checks["3_target_source_consistency"] = "PASS"

        # Check 4: Deterministic ordering
        if ordering_violations > 0:
            errors.append(f"Rule 4 Violation: {ordering_violations} sorting inversions detected (must be strictly sorted)")
            checks["4_deterministic_ordering"] = "FAIL"
        else:
            checks["4_deterministic_ordering"] = "PASS"

        # Check 5: Duplicate pairs
        if duplicate_count > 0:
            warnings.append(f"Rule 5 Warning: {duplicate_count} duplicate candidate pairs were encountered and handled")
            checks["5_duplicate_pairs"] = "WARN"
        else:
            checks["5_duplicate_pairs"] = "PASS"

        # Check 6: Test leakage
        if test_leakage_count > 0:
            errors.append(f"Rule 6 Violation: {test_leakage_count} test entity IDs leaked into candidate universe")
            checks["6_test_leakage"] = "FAIL"
        else:
            checks["6_test_leakage"] = "PASS"

        # Check 7: Candidate count preservation
        if expected_candidate_count is not None and total_output_count != expected_candidate_count:
            if total_output_count < expected_candidate_count:
                warnings.append(
                    f"Rule 7 Warning: Output candidates ({total_output_count}) != expected ({expected_candidate_count})"
                )
                checks["7_candidate_count_preservation"] = "WARN"
            else:
                checks["7_candidate_count_preservation"] = "PASS"
        else:
            checks["7_candidate_count_preservation"] = "PASS"

        # Check 8: Metadata recorded
        if not git_commit or git_commit == "unknown":
            warnings.append("Rule 8 Warning: Git commit hash is unrecorded.")
            checks["8_git_commit"] = "WARN"
        else:
            checks["8_git_commit"] = "PASS"

        status = "FAIL" if errors else "PASS"

        metrics_summary = {
            "total_input_candidates": total_input_count,
            "total_output_candidates": total_output_count,
            "duplicate_candidates": duplicate_count,
            "invalid_s1_ids": invalid_s1_count,
            "invalid_target_ids": invalid_target_count,
            "inconsistent_sources": inconsistent_source_count,
            "test_leakage_detected": test_leakage_count,
            "ordering_violations": ordering_violations,
        }

        self.logger.info(
            "Phase 5 Validation complete: Status=%s (Errors=%d, Warnings=%d, Output Candidates=%d)",
            status,
            len(errors),
            len(warnings),
            total_output_count,
        )

        return CandidateValidationReport(
            status=status,
            checks=checks,
            errors=errors,
            warnings=warnings,
            metrics_summary=metrics_summary,
        )
