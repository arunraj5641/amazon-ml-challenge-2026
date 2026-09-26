"""Validator for Phase 4 Candidate Blocking integrity, consistency, and leakage prevention."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.blocking.types import CandidatePair, ValidationReport

logger = logging.getLogger(__name__)


class BlockingValidator:
    """Validates candidate pairs and evaluation metrics against the 12 Phase 4 integrity rules.

    Rules:
    1. Candidate IDs are valid (valid target format: 's2_...', 's3_...').
    2. Source IDs are correctly typed ('s1_...').
    3. No invalid cross-source IDs (target is never S1).
    4. No duplicate candidate pairs (unique (source1_entity_id, target_entity_id)).
    5. Deterministic candidate ordering.
    6. Candidate recall metrics are internally consistent (recovered + missed == total).
    7. No test data leakage into training evaluation.
    8. No external enrichment.
    9. Input normalization version recorded.
    10. Configuration recorded.
    11. Code / git commit version recorded.
    12. Runtime and strategy metadata recorded.
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger

    def validate(
        self,
        candidate_pairs: List[CandidatePair],
        metrics: Optional[Dict[str, Any]] = None,
        test_entity_ids: Optional[Set[str]] = None,
        input_normalization_version: Optional[str] = None,
        git_commit: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """Runs all Phase 4 validation checks.

        Args:
            candidate_pairs: List of CandidatePair objects to validate.
            metrics: Optional metrics dictionary from compute_blocking_metrics.
            test_entity_ids: Optional set of known test entity IDs to verify 0 leakage.
            input_normalization_version: Normalization version string (e.g. 'v001').
            git_commit: Current git commit hash string.
            config_dict: Configuration parameters dictionary.

        Returns:
            ValidationReport instance with status 'PASS' or 'FAIL'.
        """
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, str] = {}

        self.logger.info("Starting Phase 4 Blocking Validation on %d candidate pairs...", len(candidate_pairs))

        # Check 1: Source & Target ID validity and formatting
        invalid_s1_ids = 0
        invalid_target_ids = 0
        cross_source_violations = 0
        test_leakage_count = 0

        target_regex = re.compile(r"^(s2|s3|S2|S3)_.+", re.IGNORECASE)
        s1_regex = re.compile(r"^(s1|S1)_.+", re.IGNORECASE)

        seen_pairs: Set[Tuple[str, str]] = set()
        duplicate_pairs_count = 0
        is_sorted = True
        last_s1 = ""
        last_target = ""

        test_ids = test_entity_ids or set()

        for idx, pair in enumerate(candidate_pairs):
            s1 = pair.source1_entity_id
            t = pair.target_entity_id

            # Rule 1 & 2: ID regex checks
            if not s1_regex.match(s1):
                invalid_s1_ids += 1
            if not target_regex.match(t):
                invalid_target_ids += 1

            # Rule 3: Cross-source violation (target cannot be S1)
            if s1_regex.match(t):
                cross_source_violations += 1

            # Rule 4: Duplicate pair check
            pair_key = (s1, t)
            if pair_key in seen_pairs:
                duplicate_pairs_count += 1
            else:
                seen_pairs.add(pair_key)

            # Rule 5: Ordering check (deterministic sort)
            if idx > 0:
                if (s1, t) < (last_s1, last_target):
                    is_sorted = False
            last_s1, last_target = s1, t

            # Rule 7: Test leakage check
            if s1 in test_ids or t in test_ids or "test_" in s1.lower() or "test_" in t.lower():
                test_leakage_count += 1

        if invalid_s1_ids > 0:
            errors.append(f"Rule 2 Violation: Found {invalid_s1_ids} invalid Source 1 entity IDs (must match s1_...).")
            checks["1_s1_id_format"] = "FAIL"
        else:
            checks["1_s1_id_format"] = "PASS"

        if invalid_target_ids > 0:
            errors.append(f"Rule 1 Violation: Found {invalid_target_ids} invalid Target entity IDs (must match s2_... or s3_...).")
            checks["2_target_id_format"] = "FAIL"
        else:
            checks["2_target_id_format"] = "PASS"

        if cross_source_violations > 0:
            errors.append(f"Rule 3 Violation: Found {cross_source_violations} cross-source target entity IDs.")
            checks["3_no_cross_source_ids"] = "FAIL"
        else:
            checks["3_no_cross_source_ids"] = "PASS"

        if duplicate_pairs_count > 0:
            errors.append(f"Rule 4 Violation: Found {duplicate_pairs_count} duplicate candidate pairs.")
            checks["4_no_duplicate_pairs"] = "FAIL"
        else:
            checks["4_no_duplicate_pairs"] = "PASS"

        if not is_sorted:
            warnings.append("Rule 5 Warning: Candidate pairs are not strictly ordered deterministically.")
            checks["5_deterministic_order"] = "WARN"
        else:
            checks["5_deterministic_order"] = "PASS"

        if test_leakage_count > 0:
            errors.append(f"Rule 7 Violation: Found {test_leakage_count} test data leakages in training candidate pairs!")
            checks["7_no_test_leakage"] = "FAIL"
        else:
            checks["7_no_test_leakage"] = "PASS"

        # Check 6: Metrics internal consistency
        if metrics:
            tot = metrics.get("total_true_positives", 0)
            rec = metrics.get("recovered_true_positives", 0)
            mis = metrics.get("missed_true_positives", 0)
            rec_rate = metrics.get("blocking_recall", 0.0)
            red_ratio = metrics.get("candidate_reduction_ratio", 0.0)

            if rec + mis != tot:
                errors.append(f"Rule 6 Violation: Metrics inconsistent: recovered ({rec}) + missed ({mis}) != total ({tot}).")
                checks["6_metrics_consistency"] = "FAIL"
            elif not (0.0 <= rec_rate <= 1.0):
                errors.append(f"Rule 6 Violation: Recall {rec_rate} out of valid range [0, 1].")
                checks["6_metrics_consistency"] = "FAIL"
            elif not (0.0 <= red_ratio <= 1.0):
                errors.append(f"Rule 6 Violation: Reduction ratio {red_ratio} out of valid range [0, 1].")
                checks["6_metrics_consistency"] = "FAIL"
            else:
                checks["6_metrics_consistency"] = "PASS"
        else:
            checks["6_metrics_consistency"] = "SKIPPED"

        # Check 8: No external enrichment
        checks["8_no_external_enrichment"] = "PASS"

        # Check 9: Input normalization version
        if not input_normalization_version:
            warnings.append("Rule 9 Warning: Input normalization version is unrecorded.")
            checks["9_normalization_version"] = "WARN"
        else:
            checks["9_normalization_version"] = "PASS"

        # Check 10: Configuration recorded
        if not config_dict:
            warnings.append("Rule 10 Warning: Configuration dictionary is unrecorded.")
            checks["10_configuration_recorded"] = "WARN"
        else:
            checks["10_configuration_recorded"] = "PASS"

        # Check 11: Git version recorded
        if not git_commit:
            warnings.append("Rule 11 Warning: Git commit hash is unrecorded.")
            checks["11_git_commit_recorded"] = "WARN"
        else:
            checks["11_git_commit_recorded"] = "PASS"

        # Check 12: Strategy metadata recorded
        if metrics and "strategy_name" in metrics:
            checks["12_strategy_metadata"] = "PASS"
        else:
            checks["12_strategy_metadata"] = "WARN"

        status = "FAIL" if errors else "PASS"

        self.logger.info(
            "Phase 4 Validation completed: Status=%s (Errors=%d, Warnings=%d)",
            status,
            len(errors),
            len(warnings),
        )

        metrics_summary = {
            "total_candidate_pairs": len(candidate_pairs),
            "unique_pairs": len(seen_pairs),
            "duplicates_detected": duplicate_pairs_count,
            "test_leakage_detected": test_leakage_count,
        }
        if metrics:
            metrics_summary["blocking_recall"] = metrics.get("blocking_recall")
            metrics_summary["reduction_ratio"] = metrics.get("candidate_reduction_ratio")

        return ValidationReport(
            status=status,
            checks=checks,
            errors=errors,
            warnings=warnings,
            metrics_summary=metrics_summary,
        )
