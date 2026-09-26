"""Metric computations for blocking recall, candidate reduction, and candidate distribution."""

from collections import defaultdict
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.blocking.types import CandidatePair, StrategyMetrics


def calculate_percentile(values: List[int], p: float) -> float:
    """Calculates percentile p (0-100) deterministically from a list of ints."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, p))


def compute_blocking_metrics(
    candidate_pairs: List[CandidatePair],
    authoritative_positives: Set[Tuple[str, str]],
    evaluated_s1_ids: Set[str],
    total_target_pool_size: int,
    s1_country_map: Optional[Dict[str, str]] = None,
    s1_gt_matches_map: Optional[Dict[str, Set[str]]] = None,
    runtime_seconds: float = 0.0,
    peak_memory_mb: float = 0.0,
    strategy_name: str = "evaluated_strategy",
) -> Dict[str, Any]:
    """Computes comprehensive blocking metrics against authoritative ground truth.

    Args:
        candidate_pairs: Generated candidate pairs for evaluated S1 entities.
        authoritative_positives: Authoritative true positive set `(s1_id, target_id)`.
        evaluated_s1_ids: Set of Source 1 entity IDs that were evaluated.
        total_target_pool_size: Total available target entities (|S2| + |S3|).
        s1_country_map: Optional mapping of s1_id -> country_normalized.
        s1_gt_matches_map: Optional mapping of s1_id -> set of true matching target IDs.
        runtime_seconds: Elapsed runtime in seconds.
        peak_memory_mb: Estimated peak memory usage in MB.
        strategy_name: Identifier name for the strategy.

    Returns:
        Structured dictionary containing metrics, distributions, per-source and per-country breakdowns.
    """
    total_s1 = len(evaluated_s1_ids)

    # 1. Candidate lookup set
    generated_cand_set: Set[Tuple[str, str]] = {
        (p.source1_entity_id, p.target_entity_id) for p in candidate_pairs
    }

    # 2. Count candidates per S1
    cand_counts_by_s1: Dict[str, int] = defaultdict(int)
    for p in candidate_pairs:
        cand_counts_by_s1[p.source1_entity_id] += 1

    counts_list = [cand_counts_by_s1[s1] for s1 in evaluated_s1_ids]
    if not counts_list:
        counts_list = [0]

    avg_cands = float(np.mean(counts_list)) if counts_list else 0.0
    median_cands = float(np.median(counts_list)) if counts_list else 0.0
    min_cands = int(min(counts_list)) if counts_list else 0
    max_cands = int(max(counts_list)) if counts_list else 0
    p90_cands = calculate_percentile(counts_list, 90)
    p95_cands = calculate_percentile(counts_list, 95)
    p99_cands = calculate_percentile(counts_list, 99)

    # 3. Overall True Positive Recovery
    # Filter authoritative positives to only those where S1 was evaluated
    relevant_positives = {
        (s1, t) for (s1, t) in authoritative_positives if s1 in evaluated_s1_ids
    }
    total_relevant_pos = len(relevant_positives)

    recovered_positives = relevant_positives.intersection(generated_cand_set)
    num_recovered = len(recovered_positives)
    num_missed = total_relevant_pos - num_recovered

    recall = (num_recovered / total_relevant_pos) if total_relevant_pos > 0 else 1.0

    # 4. Cartesian Comparison & Reduction Ratio
    total_candidate_count = len(generated_cand_set)
    total_cartesian_pairs = total_s1 * total_target_pool_size
    if total_cartesian_pairs > 0:
        reduction_ratio = 1.0 - (total_candidate_count / total_cartesian_pairs)
    else:
        reduction_ratio = 1.0

    # 5. Per-Source Recall (S2 vs S3)
    s2_positives = {(s1, t) for (s1, t) in relevant_positives if t.startswith("s2_") or t.startswith("S2_")}
    s3_positives = {(s1, t) for (s1, t) in relevant_positives if t.startswith("s3_") or t.startswith("S3_")}

    s2_recovered = s2_positives.intersection(generated_cand_set)
    s3_recovered = s3_positives.intersection(generated_cand_set)

    s2_recall = (len(s2_recovered) / len(s2_positives)) if s2_positives else 1.0
    s3_recall = (len(s3_recovered) / len(s3_positives)) if s3_positives else 1.0

    # 6. Per-Country Recall Breakdown
    country_map = s1_country_map or {}
    country_positives: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for s1, t in relevant_positives:
        c = country_map.get(s1, "unknown")
        country_positives[c].add((s1, t))

    per_country_recall: Dict[str, Dict[str, Any]] = {}
    for c, pos_set in sorted(country_positives.items()):
        rec_set = pos_set.intersection(generated_cand_set)
        c_recall = len(rec_set) / len(pos_set) if pos_set else 1.0
        per_country_recall[c] = {
            "total_positives": len(pos_set),
            "recovered_positives": len(rec_set),
            "recall": round(c_recall, 4),
        }

    # 7. Zero-Match & Multi-Match Behavior
    gt_matches = s1_gt_matches_map or {}
    zero_match_s1_list: List[str] = []
    single_match_s1_list: List[str] = []
    multi_match_s1_list: List[str] = []

    for s1 in evaluated_s1_ids:
        matches = gt_matches.get(s1, set())
        if len(matches) == 0:
            zero_match_s1_list.append(s1)
        elif len(matches) == 1:
            single_match_s1_list.append(s1)
        else:
            multi_match_s1_list.append(s1)

    # Zero-match candidate generation statistics
    zero_match_cands = [cand_counts_by_s1[s1] for s1 in zero_match_s1_list]
    zero_match_stats = {
        "count": len(zero_match_s1_list),
        "avg_candidates_generated": round(float(np.mean(zero_match_cands)), 2) if zero_match_cands else 0.0,
        "median_candidates_generated": round(float(np.median(zero_match_cands)), 2) if zero_match_cands else 0.0,
        "max_candidates_generated": int(max(zero_match_cands)) if zero_match_cands else 0,
    }

    # Multi-match recovery statistics
    multi_match_pos = 0
    multi_match_rec = 0
    full_recovery_count = 0
    for s1 in multi_match_s1_list:
        targets = gt_matches.get(s1, set())
        multi_match_pos += len(targets)
        matched_targets = {t for t in targets if (s1, t) in generated_cand_set}
        multi_match_rec += len(matched_targets)
        if len(matched_targets) == len(targets):
            full_recovery_count += 1

    multi_match_stats = {
        "count": len(multi_match_s1_list),
        "total_matches": multi_match_pos,
        "recovered_matches": multi_match_rec,
        "recall": round((multi_match_rec / multi_match_pos) if multi_match_pos > 0 else 1.0, 4),
        "entities_with_100pct_recovery": full_recovery_count,
        "full_recovery_rate": round((full_recovery_count / len(multi_match_s1_list)) if multi_match_s1_list else 1.0, 4),
    }

    return {
        "strategy_name": strategy_name,
        "evaluated_s1_count": total_s1,
        "total_target_pool_size": total_target_pool_size,
        "total_cartesian_pairs": total_cartesian_pairs,
        "total_candidate_pairs": total_candidate_count,
        "candidate_reduction_ratio": round(reduction_ratio, 6),
        "blocking_recall": round(recall, 6),
        "total_true_positives": total_relevant_pos,
        "recovered_true_positives": num_recovered,
        "missed_true_positives": num_missed,
        "candidates_per_s1": {
            "average": round(avg_cands, 2),
            "median": round(median_cands, 2),
            "min": min_cands,
            "max": max_cands,
            "p90": round(p90_cands, 2),
            "p95": round(p95_cands, 2),
            "p99": round(p99_cands, 2),
        },
        "per_source_recall": {
            "s2_recall": round(s2_recall, 6),
            "s2_total_positives": len(s2_positives),
            "s2_recovered_positives": len(s2_recovered),
            "s3_recall": round(s3_recall, 6),
            "s3_total_positives": len(s3_positives),
            "s3_recovered_positives": len(s3_recovered),
        },
        "per_country_recall": per_country_recall,
        "zero_match_s1_behavior": zero_match_stats,
        "multi_match_s1_behavior": multi_match_stats,
        "runtime_seconds": round(runtime_seconds, 4),
        "peak_memory_mb": round(peak_memory_mb, 2),
    }
