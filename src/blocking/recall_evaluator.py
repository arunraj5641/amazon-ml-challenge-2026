"""Standalone, memory-safe Blocking Recall Evaluator for Phase 4.

Streams ~500M candidate pairs against sorted Phase 3 ground-truth positives
using a deterministic O(1)-RAM two-pointer comparison.
"""

from datetime import datetime, timezone
import gzip
import io
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from src.data.data_source import S3DataSource, create_data_source


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, falling back to 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


logger = logging.getLogger(__name__)


def stream_text_lines(
    path_or_uri: Union[str, Path],
    region: Optional[str] = None,
    s3_client: Optional[Any] = None,
) -> Iterator[str]:
    """Streams text lines one by one from local file or S3 object without loading whole file into RAM.

    Supports both uncompressed and gzip-compressed (.gz) sources.
    """
    path_str = str(path_or_uri).strip()
    is_gz = path_str.endswith(".gz")

    if path_str.startswith("s3://"):
        data_source = create_data_source(path_str, region=region, s3_client=s3_client)
        if not isinstance(data_source, S3DataSource):
            raise ValueError(f"Expected S3DataSource for {path_str}")

        bucket = data_source.bucket
        key = data_source.prefix
        resp = data_source.client.get_object(Bucket=bucket, Key=key)
        raw_body = resp["Body"]

        if is_gz:
            gz_stream = gzip.GzipFile(fileobj=raw_body, mode="rb")
            text_stream = io.TextIOWrapper(gz_stream, encoding="utf-8")
        else:
            text_stream = io.TextIOWrapper(raw_body, encoding="utf-8")

        try:
            for line in text_stream:
                yield line
        finally:
            text_stream.close()
    else:
        local_p = Path(path_str).resolve()
        if not local_p.is_file():
            raise FileNotFoundError(f"Candidate file not found: {local_p}")

        if is_gz:
            with gzip.open(local_p, "rt", encoding="utf-8") as f:
                for line in f:
                    yield line
        else:
            with open(local_p, "r", encoding="utf-8") as f:
                for line in f:
                    yield line


def load_authoritative_positives(
    gt_path_or_uri: Union[str, Path],
    region: Optional[str] = None,
    s3_client: Optional[Any] = None,
) -> List[Tuple[str, str, str]]:
    """Loads authoritative ground-truth positive pairs from Phase 3 or raw GT.

    Accepts:
    1. Phase 3 training_pairs.tsv (filtered where label == 1 or pair_type == 'positive')
    2. Raw train_ground_truth.tsv (source1_entity_id, matched_entity_ids)

    Returns:
        List of unique (source1_entity_id, target_entity_id, target_source) tuples
        sorted deterministically by (source1_entity_id, target_entity_id).
    """
    logger.info("Loading ground-truth positives from %s...", gt_path_or_uri)
    positives_set: Set[Tuple[str, str, str]] = set()

    line_iter = stream_text_lines(gt_path_or_uri, region=region, s3_client=s3_client)
    first_line = next(line_iter, None)
    if first_line is None:
        return []

    header = [h.strip() for h in first_line.rstrip("\r\n").split("\t")]

    # Check format: Phase 3 training pairs vs Raw train_ground_truth
    is_training_pairs = "label" in header and "target_entity_id" in header
    is_raw_gt = "matched_entity_ids" in header and "source1_entity_id" in header

    col_s1_idx: int = 0
    col_target_idx: int = -1
    col_matches_idx: int = -1
    col_src_idx: int = -1
    col_label_idx: int = -1

    if is_training_pairs:
        col_s1_idx = header.index("source1_entity_id")
        col_target_idx = header.index("target_entity_id")
        col_label_idx = header.index("label")
        if "target_source" in header:
            col_src_idx = header.index("target_source")
    elif is_raw_gt:
        col_s1_idx = header.index("source1_entity_id")
        col_matches_idx = header.index("matched_entity_ids")
    else:
        col_s1_idx = 0
        col_target_idx = 1
        if "label" in header:
            col_label_idx = header.index("label")

    for line in line_iter:
        parts = line.rstrip("\r\n").split("\t")
        if not parts or len(parts) <= col_s1_idx:
            continue
        s1 = parts[col_s1_idx].strip()
        if not s1:
            continue

        if is_training_pairs or (col_label_idx >= 0 and len(parts) > col_label_idx):
            label_val = parts[col_label_idx].strip()
            if label_val not in ("1", "1.0", "positive"):
                continue
            target_id = parts[col_target_idx].strip() if len(parts) > col_target_idx else ""
            if not target_id:
                continue
            if col_src_idx >= 0 and len(parts) > col_src_idx and parts[col_src_idx].strip():
                src = parts[col_src_idx].strip()
            else:
                t_lower = target_id.lower()
                src = "Source 2" if (t_lower.startswith("s2_") or t_lower.startswith("s2-") or "_s2_" in t_lower) else "Source 3"
            positives_set.add((s1, target_id, src))

        elif is_raw_gt:
            if len(parts) <= col_matches_idx:
                continue
            raw_matches = parts[col_matches_idx].strip()
            if not raw_matches or raw_matches.lower() in ("nan", "null", "none", "<na>"):
                continue
            tokens = [t.strip() for t in raw_matches.split(",") if t.strip()]
            for target_id in tokens:
                t_lower = target_id.lower()
                src = "Source 2" if (t_lower.startswith("s2_") or t_lower.startswith("s2-") or "_s2_" in t_lower) else "Source 3"
                positives_set.add((s1, target_id, src))
        else:
            # Simple 2-column positive pair format
            target_id = parts[col_target_idx].strip()
            t_lower = target_id.lower()
            src = "Source 2" if (t_lower.startswith("s2_") or t_lower.startswith("s2-") or "_s2_" in t_lower) else "Source 3"
            positives_set.add((s1, target_id, src))

    # Sort deterministically by (s1_id, target_id)
    sorted_positives = sorted(positives_set, key=lambda p: (p[0], p[1]))
    logger.info("Loaded and sorted %d unique authoritative positive pairs.", len(sorted_positives))
    return sorted_positives


class BlockingRecallEvaluator:
    """Evaluates blocking recall on full-scale candidate pairs in a memory-safe streaming fashion."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logger

    def evaluate(
        self,
        candidate_lines: Iterator[str],
        sorted_ground_truth: List[Tuple[str, str, str]],
        missed_output_tsv_path: Optional[Union[str, Path]] = None,
        total_s1_pool_size: Optional[int] = 2_206_821,
        total_target_pool_size: Optional[int] = 10_320_219,
    ) -> Dict[str, Any]:
        """Runs streaming two-pointer recall evaluation.

        Args:
            candidate_lines: Iterator yielding lines from candidate_pairs.tsv.
            sorted_ground_truth: List of (s1, target, source) sorted by (s1, target).
            missed_output_tsv_path: Local path to write missed_positive_pairs.tsv.gz.
            total_s1_pool_size: Source 1 pool count for candidate reduction calculation.
            total_target_pool_size: Target pool count (|S2| + |S3|) for candidate reduction.

        Returns:
            Dictionary containing metrics and integrity validation results.
        """
        start_time = time.time()
        n_gt = len(sorted_ground_truth)
        gt_idx = 0

        total_candidates_processed = 0
        recovered_count = 0
        missed_count = 0

        # Target source breakdowns
        s2_total = sum(1 for _, _, src in sorted_ground_truth if src == "Source 2")
        s3_total = n_gt - s2_total
        s2_recovered = 0
        s3_recovered = 0
        s2_missed = 0
        s3_missed = 0

        # S1-level tracking
        cur_s1: Optional[str] = None
        s1_total_positives = 0
        s1_recovered_positives = 0
        s1_with_positives_count = 0
        s1_zero_recovery_count = 0
        s1_full_recovery_count = 0
        s1_partial_recovery_count = 0
        s1_recovery_rate_sum = 0.0

        def _finish_s1_entity():
            nonlocal s1_with_positives_count, s1_zero_recovery_count
            nonlocal s1_full_recovery_count, s1_partial_recovery_count, s1_recovery_rate_sum
            if cur_s1 is None or s1_total_positives == 0:
                return
            s1_with_positives_count += 1
            rate = s1_recovered_positives / s1_total_positives
            s1_recovery_rate_sum += rate
            if s1_recovered_positives == 0:
                s1_zero_recovery_count += 1
            elif s1_recovered_positives == s1_total_positives:
                s1_full_recovery_count += 1
            else:
                s1_partial_recovery_count += 1

        # Setup missed output handle
        missed_handle = None
        if missed_output_tsv_path:
            p = Path(missed_output_tsv_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            if str(p).endswith(".gz"):
                missed_handle = gzip.open(p, "wt", encoding="utf-8")
            else:
                missed_handle = open(p, "wt", encoding="utf-8")
            missed_handle.write("source1_entity_id\ttarget_entity_id\ttarget_source\treason\n")

        def _record_missed(gt_tuple: Tuple[str, str, str]):
            nonlocal missed_count, s2_missed, s3_missed
            missed_count += 1
            s1_id, t_id, src = gt_tuple
            if src == "Source 2":
                s2_missed += 1
            else:
                s3_missed += 1
            if missed_handle:
                missed_handle.write(f"{s1_id}\t{t_id}\t{src}\tnot_blocked\n")

        def _advance_s1_tracker(s1_id: str, is_rec: bool):
            nonlocal cur_s1, s1_total_positives, s1_recovered_positives
            if s1_id != cur_s1:
                _finish_s1_entity()
                cur_s1 = s1_id
                s1_total_positives = 0
                s1_recovered_positives = 0
            s1_total_positives += 1
            if is_rec:
                s1_recovered_positives += 1

        last_cand_pair: Optional[Tuple[str, str]] = None
        sorted_order_valid = True

        try:
            for line in candidate_lines:
                # Fast header check
                if total_candidates_processed == 0 and ("entity_id" in line or "source1" in line):
                    continue

                idx1 = line.find("\t")
                if idx1 == -1:
                    continue
                idx2 = line.find("\t", idx1 + 1)
                if idx2 == -1:
                    cand_s1 = line[:idx1].strip()
                    cand_target = line[idx1 + 1 :].rstrip("\r\n").strip()
                else:
                    cand_s1 = line[:idx1].strip()
                    cand_target = line[idx1 + 1 : idx2].strip()

                cand_pair = (cand_s1, cand_target)
                total_candidates_processed += 1

                # Check deterministic sort order & skip duplicates
                if last_cand_pair is not None:
                    if cand_pair < last_cand_pair:
                        sorted_order_valid = False
                    elif cand_pair == last_cand_pair:
                        continue  # deduplicate repeated candidate pair
                last_cand_pair = cand_pair

                # Fast 2-pointer scan
                while gt_idx < n_gt and (sorted_ground_truth[gt_idx][0], sorted_ground_truth[gt_idx][1]) < cand_pair:
                    gt_tuple = sorted_ground_truth[gt_idx]
                    _record_missed(gt_tuple)
                    _advance_s1_tracker(gt_tuple[0], is_rec=False)
                    gt_idx += 1

                if gt_idx < n_gt and (sorted_ground_truth[gt_idx][0], sorted_ground_truth[gt_idx][1]) == cand_pair:
                    gt_tuple = sorted_ground_truth[gt_idx]
                    recovered_count += 1
                    if gt_tuple[2] == "Source 2":
                        s2_recovered += 1
                    else:
                        s3_recovered += 1
                    _advance_s1_tracker(gt_tuple[0], is_rec=True)
                    gt_idx += 1

            # Candidate stream finished: any remaining GT pairs are missed
            while gt_idx < n_gt:
                gt_tuple = sorted_ground_truth[gt_idx]
                _record_missed(gt_tuple)
                _advance_s1_tracker(gt_tuple[0], is_rec=False)
                gt_idx += 1

            _finish_s1_entity()

        finally:
            if missed_handle:
                missed_handle.close()

        elapsed_sec = time.time() - start_time
        overall_recall = (recovered_count / n_gt) if n_gt > 0 else 0.0
        s2_recall = (s2_recovered / s2_total) if s2_total > 0 else 0.0
        s3_recall = (s3_recovered / s3_total) if s3_total > 0 else 0.0
        macro_avg_s1_recall = (s1_recovery_rate_sum / s1_with_positives_count) if s1_with_positives_count > 0 else 0.0

        # Candidate reduction ratio
        candidate_reduction = None
        if total_s1_pool_size and total_target_pool_size:
            cartesian_product = total_s1_pool_size * total_target_pool_size
            candidate_reduction = 1.0 - (total_candidates_processed / cartesian_product)

        # Integrity assertions
        integrity_errors = []
        if recovered_count + missed_count != n_gt:
            integrity_errors.append(
                f"Math integrity violated: recovered ({recovered_count}) + missed ({missed_count}) != total ({n_gt})"
            )
        if s2_recovered + s2_missed != s2_total:
            integrity_errors.append(f"S2 count mismatch: {s2_recovered} + {s2_missed} != {s2_total}")
        if s3_recovered + s3_missed != s3_total:
            integrity_errors.append(f"S3 count mismatch: {s3_recovered} + {s3_missed} != {s3_total}")
        if not (0.0 <= overall_recall <= 1.0):
            integrity_errors.append(f"Recall out of range [0, 1]: {overall_recall}")
        if not sorted_order_valid:
            integrity_errors.append("Candidate pairs were not strictly sorted in ascending order")

        report = {
            "overall": {
                "total_positive_pairs": n_gt,
                "recovered_positive_pairs": recovered_count,
                "missed_positive_pairs": missed_count,
                "blocking_recall": round(overall_recall, 6),
                "candidate_pairs_evaluated": total_candidates_processed,
                "candidate_reduction_ratio": round(candidate_reduction, 8) if candidate_reduction is not None else None,
            },
            "per_target_source": {
                "source2": {
                    "total_positives": s2_total,
                    "recovered": s2_recovered,
                    "missed": s2_missed,
                    "recall": round(s2_recall, 6),
                },
                "source3": {
                    "total_positives": s3_total,
                    "recovered": s3_recovered,
                    "missed": s3_missed,
                    "recall": round(s3_recall, 6),
                },
            },
            "s1_level_metrics": {
                "s1_entities_with_positives": s1_with_positives_count,
                "zero_recovery_s1_count": s1_zero_recovery_count,
                "full_recovery_s1_count": s1_full_recovery_count,
                "partial_recovery_s1_count": s1_partial_recovery_count,
                "macro_avg_s1_recall": round(macro_avg_s1_recall, 6),
            },
            "integrity_validation": {
                "status": "PASS" if not integrity_errors else "FAIL",
                "errors": integrity_errors,
                "deterministic_sort_verified": sorted_order_valid,
            },
            "execution": {
                "runtime_seconds": round(elapsed_sec, 2),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        }

        self.logger.info(
            "Evaluation Complete: Recall=%.4f (%d/%d recovered), Missed=%d, S2 Recall=%.4f, S3 Recall=%.4f in %.1fs",
            overall_recall,
            recovered_count,
            n_gt,
            missed_count,
            s2_recall,
            s3_recall,
            elapsed_sec,
        )

        return report
