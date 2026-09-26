"""Phase 5 Streaming Candidate Pipeline.

Finalizes the Phase 4 candidate universe for downstream Phase 6 feature engineering:
- Streams ~500M candidate rows with strict O(1) RAM.
- Enforces deterministic ordering and verifies uniqueness.
- Validates ID schemas, source labels, and zero test leakage.
- Emits finalized candidate_pairs.tsv and comprehensive JSON reports.
"""

from collections import defaultdict
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from src.blocking.io import save_candidate_pairs_file, save_json_artifact
from src.blocking.recall_evaluator import stream_text_lines
from src.candidates.config import (
    BASELINE_CANDIDATE_PAIRS,
    BASELINE_S1_RECORDS,
    BASELINE_TARGET_POOL,
    CandidatePipelineConfig,
)
from src.candidates.validator import (
    CandidateValidationReport,
    CandidateValidator,
    S1_ID_PATTERN,
    S2_ID_PATTERN,
    S3_ID_PATTERN,
    TARGET_ID_PATTERN,
)
from src.data.data_source import DataSource, LocalDataSource, S3DataSource, create_data_source

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, falling back to 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


class CandidatePipeline:
    """Orchestrates Phase 5 candidate finalization and validation."""

    def __init__(
        self,
        config: Optional[CandidatePipelineConfig] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.config = config or CandidatePipelineConfig()
        self.logger = logger_instance or logger
        self.validator = CandidateValidator(logger_instance=self.logger)

    def finalize(
        self,
        input_lines: Iterator[str],
        output_file_path: Union[str, Path],
        test_entity_ids: Optional[Set[str]] = None,
        git_commit: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], CandidateValidationReport]:
        """Streams candidate pairs, validates integrity, and writes finalized candidate_pairs.tsv.

        Args:
            input_lines: Iterator yielding lines from Phase 4 candidate_pairs.tsv.
            output_file_path: Local path where finalized candidate_pairs.tsv is written.
            test_entity_ids: Optional set of known test entity IDs for leakage verification.
            git_commit: Optional git commit hash.

        Returns:
            Tuple of (stats_dict, validation_report).
        """
        start_time = time.time()
        out_p = Path(output_file_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        git_hash = git_commit or get_git_commit_hash()
        test_ids_set = test_entity_ids or set()

        total_input_count = 0
        total_output_count = 0
        duplicate_count = 0
        invalid_s1_count = 0
        invalid_target_count = 0
        inconsistent_source_count = 0
        test_leakage_count = 0
        ordering_violations = 0

        s2_count = 0
        s3_count = 0
        strategy_counts: Dict[str, int] = defaultdict(int)

        # S1 entity tracking (O(1) RAM relying on primary sort by S1)
        unique_s1_count = 0
        cur_s1: Optional[str] = None
        cur_s1_degree = 0
        min_s1_degree = float("inf")
        max_s1_degree = 0

        last_cand_pair: Optional[Tuple[str, str]] = None

        with open(out_p, "wt", encoding="utf-8") as out_f:
            for line in input_lines:
                # Fast header check
                if total_input_count == 0 and ("entity_id" in line or "source1" in line):
                    out_f.write("source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n")
                    continue

                total_input_count += 1
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue

                parts = stripped.split("\t")
                if len(parts) < 2:
                    continue

                cand_s1 = parts[0].strip()
                cand_target = parts[1].strip()
                cand_source = parts[2].strip() if len(parts) > 2 else ""
                cand_strats = parts[3].strip() if len(parts) > 3 else ""

                # Rule 1 & 2: ID Format validation
                if not S1_ID_PATTERN.match(cand_s1):
                    invalid_s1_count += 1
                if not TARGET_ID_PATTERN.match(cand_target):
                    invalid_target_count += 1

                # Rule 3: Source label consistency
                is_s2 = bool(S2_ID_PATTERN.match(cand_target))
                expected_src = "Source 2" if is_s2 else "Source 3"
                if cand_source != expected_src:
                    inconsistent_source_count += 1
                    cand_source = expected_src  # Normalize

                # Rule 4 & 5: Deterministic ordering & Duplicate detection
                cand_pair = (cand_s1, cand_target)
                if last_cand_pair is not None:
                    if cand_pair < last_cand_pair:
                        ordering_violations += 1
                    elif cand_pair == last_cand_pair:
                        duplicate_count += 1
                        if self.config.allow_deduplication:
                            continue  # Drop redundant candidate

                last_cand_pair = cand_pair

                # Rule 6: Test Leakage check
                if test_ids_set and (cand_s1 in test_ids_set or cand_target in test_ids_set):
                    test_leakage_count += 1
                    continue

                # Write finalized row
                out_f.write(f"{cand_s1}\t{cand_target}\t{cand_source}\t{cand_strats}\n")
                total_output_count += 1

                # Update per-target counts
                if is_s2:
                    s2_count += 1
                else:
                    s3_count += 1

                # Update strategy frequencies
                if cand_strats:
                    for strat in cand_strats.split(","):
                        s_name = strat.strip()
                        if s_name:
                            strategy_counts[s_name] += 1

                # S1 tracking
                if cand_s1 != cur_s1:
                    if cur_s1 is not None:
                        unique_s1_count += 1
                        if cur_s1_degree < min_s1_degree:
                            min_s1_degree = cur_s1_degree
                        if cur_s1_degree > max_s1_degree:
                            max_s1_degree = cur_s1_degree
                    cur_s1 = cand_s1
                    cur_s1_degree = 0

                cur_s1_degree += 1

            # Final S1 flush
            if cur_s1 is not None:
                unique_s1_count += 1
                if cur_s1_degree < min_s1_degree:
                    min_s1_degree = cur_s1_degree
                if cur_s1_degree > max_s1_degree:
                    max_s1_degree = cur_s1_degree

        elapsed = time.time() - start_time
        if min_s1_degree == float("inf"):
            min_s1_degree = 0

        avg_s1_degree = (total_output_count / unique_s1_count) if unique_s1_count > 0 else 0.0

        # Build validation report
        val_report = self.validator.build_report(
            total_input_count=total_input_count,
            total_output_count=total_output_count,
            duplicate_count=duplicate_count,
            invalid_s1_count=invalid_s1_count,
            invalid_target_count=invalid_target_count,
            inconsistent_source_count=inconsistent_source_count,
            test_leakage_count=test_leakage_count,
            ordering_violations=ordering_violations,
            expected_candidate_count=self.config.expected_candidate_count,
            git_commit=git_hash,
            config_dict=self.config.to_dict(),
        )

        stats = {
            "total_input_candidates": total_input_count,
            "total_output_candidates": total_output_count,
            "unique_s1_entities": unique_s1_count,
            "duplicate_candidates_removed": duplicate_count,
            "source_distribution": {
                "source2_candidates": s2_count,
                "source3_candidates": s3_count,
                "source2_ratio": round(s2_count / total_output_count, 6) if total_output_count > 0 else 0.0,
                "source3_ratio": round(s3_count / total_output_count, 6) if total_output_count > 0 else 0.0,
            },
            "s1_candidate_distribution": {
                "min_candidates_per_s1": min_s1_degree,
                "max_candidates_per_s1": max_s1_degree,
                "avg_candidates_per_s1": round(avg_s1_degree, 2),
            },
            "strategy_contributions": dict(strategy_counts),
            "execution": {
                "runtime_seconds": round(elapsed, 2),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        }

        self.logger.info(
            "Phase 5 Candidate Finalization Complete: Input=%d, Output=%d, S1 Entities=%d, Duplicates=%d in %.1fs",
            total_input_count,
            total_output_count,
            unique_s1_count,
            duplicate_count,
            elapsed,
        )

        return stats, val_report
