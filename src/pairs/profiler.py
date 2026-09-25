"""Ground truth profiling utilities for Amazon ML Challenge 2026 Phase 3.

Provides streaming inspection of train_ground_truth.tsv:
- Total Source 1 entities and total ground-truth rows
- Zero-match, one-match, and multi-match distribution
- Target ID breakdown (Source 2 vs Source 3)
- Unique referenced S2 and S3 entity counts
- Duplicate ground-truth references (within row and across rows)
- Invalid references and unexpected entity ID problems
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd

from src.data.data_source import DataSource
from src.utils.logger import setup_logger


@dataclass
class GroundTruthProfile:
    """Comprehensive profiling metrics for the ground truth dataset."""

    total_source1_entities: int = 0
    ground_truth_rows: int = 0
    unique_source1_entities: int = 0
    total_positive_pairs: int = 0
    zero_match_count: int = 0
    one_match_count: int = 0
    multi_match_count: int = 0
    max_matches_per_entity: int = 0
    cardinality_distribution: Dict[int, int] = field(default_factory=dict)
    target_source_breakdown: Dict[str, int] = field(default_factory=dict)
    referenced_s2_entity_counts: int = 0
    referenced_s3_entity_counts: int = 0
    total_unique_targets_referenced: int = 0
    duplicate_references_within_row: int = 0
    duplicate_source1_rows: int = 0
    duplicate_ground_truth_references: int = 0
    invalid_references: List[str] = field(default_factory=list)
    invalid_references_count: int = 0
    unexpected_entity_id_problems: List[str] = field(default_factory=list)
    unexpected_entity_id_problems_count: int = 0
    sample_positive_pairs: List[Tuple[str, str, str]] = field(default_factory=list)
    sample_zero_match_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        total = self.total_source1_entities
        card_pct = {}
        for k, v in self.cardinality_distribution.items():
            pct = round((v / total * 100.0), 2) if total > 0 else 0.0
            card_pct[str(k)] = {"count": v, "percentage": pct}
        d["cardinality_details"] = card_pct
        return d


class GroundTruthProfiler:
    """Streams and profiles ground truth files from local or S3 data sources."""

    def __init__(
        self,
        data_source: DataSource,
        ground_truth_rel_path: str = "train/train_ground_truth.tsv",
        chunksize: int = 100000,
        logger: Optional[logging.Logger] = None,
    ):
        self.data_source = data_source
        self.gt_path = ground_truth_rel_path
        self.chunksize = chunksize
        self.logger = logger or setup_logger(name="gt_profiler")

    def profile(self, sample_size_examples: int = 10) -> GroundTruthProfile:
        """Executes a full streaming profile of the ground truth dataset."""
        self.logger.info("Profiling ground truth from: %s", self.data_source.get_uri(self.gt_path))

        profile = GroundTruthProfile()
        cardinality_counter: Counter[int] = Counter()
        target_source_counter: Counter[str] = Counter()

        seen_s1: Set[str] = set()
        seen_s2_targets: Set[str] = set()
        seen_s3_targets: Set[str] = set()
        seen_all_pairs: Set[Tuple[str, str]] = set()

        invalid_refs_list: List[str] = []
        id_problems_list: List[str] = []

        total_rows = 0
        total_pos = 0
        duplicate_ref_within_row = 0
        duplicate_pairs_across_rows = 0
        duplicate_s1_rows = 0
        max_matches = 0

        chunk_iter = self.data_source.read_chunks(
            rel_path=self.gt_path,
            sep="\t",
            chunksize=self.chunksize,
        )

        for chunk_idx, chunk in enumerate(chunk_iter):
            for _, row in chunk.iterrows():
                total_rows += 1
                raw_s1 = row.get("source1_entity_id")

                # Check S1 ID integrity
                if raw_s1 is None or pd.isna(raw_s1) or not str(raw_s1).strip():
                    id_problems_list.append(f"Row {total_rows}: Missing or null source1_entity_id")
                    s1_id = f"INVALID_ROW_{total_rows}"
                else:
                    s1_id = str(raw_s1).strip()
                    if s1_id in seen_s1:
                        duplicate_s1_rows += 1
                    seen_s1.add(s1_id)

                raw_matches = row.get("matched_entity_ids")

                # Check for 0-match singletons
                if (
                    raw_matches is None
                    or pd.isna(raw_matches)
                    or not str(raw_matches).strip()
                    or str(raw_matches).strip().lower() in ("nan", "null", "none", "<na>")
                ):
                    cardinality_counter[0] += 1
                    if len(profile.sample_zero_match_ids) < sample_size_examples:
                        profile.sample_zero_match_ids.append(s1_id)
                    continue

                raw_tokens = str(raw_matches).split(",")
                valid_tokens: List[str] = []
                seen_in_this_row: Set[str] = set()

                for tok in raw_tokens:
                    t = tok.strip()
                    if not t:
                        invalid_refs_list.append(f"Empty reference token in row {total_rows} (S1: {s1_id})")
                        continue

                    # Check within-row duplicate
                    if t in seen_in_this_row:
                        duplicate_ref_within_row += 1
                    else:
                        seen_in_this_row.add(t)

                    # Check across-row duplicate pairs
                    pair_tuple = (s1_id, t)
                    if pair_tuple in seen_all_pairs:
                        duplicate_pairs_across_rows += 1
                    else:
                        seen_all_pairs.add(pair_tuple)

                    # Determine target source and check validity
                    t_lower = t.lower()
                    if t_lower.startswith("s2-") or t_lower.startswith("s2_") or "_s2_" in t_lower:
                        target_source = "Source 2"
                        seen_s2_targets.add(t)
                    elif t_lower.startswith("s3-") or t_lower.startswith("s3_") or "_s3_" in t_lower:
                        target_source = "Source 3"
                        seen_s3_targets.add(t)
                    else:
                        target_source = "Unknown"
                        invalid_refs_list.append(
                            f"Unrecognized target prefix '{t}' in row {total_rows} (S1: {s1_id})"
                        )

                    target_source_counter[target_source] += 1
                    valid_tokens.append(t)

                    if len(profile.sample_positive_pairs) < sample_size_examples:
                        profile.sample_positive_pairs.append((s1_id, t, target_source))

                num_matches = len(valid_tokens)
                cardinality_counter[num_matches] += 1
                total_pos += num_matches

                if num_matches > max_matches:
                    max_matches = num_matches

        profile.ground_truth_rows = total_rows
        profile.total_source1_entities = total_rows
        profile.unique_source1_entities = len(seen_s1)
        profile.total_positive_pairs = total_pos
        profile.zero_match_count = cardinality_counter[0]
        profile.one_match_count = cardinality_counter[1]
        profile.multi_match_count = sum(v for k, v in cardinality_counter.items() if k > 1)
        profile.max_matches_per_entity = max_matches
        profile.cardinality_distribution = dict(sorted(cardinality_counter.items()))
        profile.target_source_breakdown = dict(target_source_counter)
        profile.referenced_s2_entity_counts = len(seen_s2_targets)
        profile.referenced_s3_entity_counts = len(seen_s3_targets)
        profile.total_unique_targets_referenced = len(seen_s2_targets | seen_s3_targets)
        profile.duplicate_references_within_row = duplicate_ref_within_row
        profile.duplicate_source1_rows = duplicate_s1_rows
        profile.duplicate_ground_truth_references = duplicate_ref_within_row + duplicate_pairs_across_rows
        profile.invalid_references_count = len(invalid_refs_list)
        profile.invalid_references = invalid_refs_list[:20]
        profile.unexpected_entity_id_problems_count = len(id_problems_list)
        profile.unexpected_entity_id_problems = id_problems_list[:20]

        self.logger.info(
            "Ground Truth Profile Complete: %d S1 rows (%d unique) | %d Positive Pairs | "
            "Referenced S2: %d | Referenced S3: %d | Zero-matches: %d (%.2f%%) | Multi-matches: %d | Invalid refs: %d",
            profile.ground_truth_rows,
            profile.unique_source1_entities,
            profile.total_positive_pairs,
            profile.referenced_s2_entity_counts,
            profile.referenced_s3_entity_counts,
            profile.zero_match_count,
            (profile.zero_match_count / profile.total_source1_entities * 100.0)
            if profile.total_source1_entities > 0
            else 0.0,
            profile.multi_match_count,
            profile.invalid_references_count,
        )

        return profile
