"""Memory-conscious, frequency-guarded blocking indexes for target entity pools."""

from collections import defaultdict
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.normalization.normalizer import normalize_country

logger = logging.getLogger(__name__)


class BlockingIndex:
    """In-memory inverted index mapping blocking keys to target entity IDs.

    Features:
    - Compact representation: stores only blocking keys, posting lists of IDs, and target metadata.
    - High-frequency key suppression: blocks with postings exceeding threshold are suppressed
      to prevent memory exhaustion and candidate explosion.
    - Source isolation & traceability: records target source ("Source 2" / "Source 3") and country.
    - Determinism: preserves deterministic traversal order.
    """

    def __init__(self, name: str = "default_index"):
        self.name = name
        self._posting_lists: Dict[str, List[str]] = defaultdict(list)
        self._target_sources: Dict[str, str] = {}
        self._target_countries: Dict[str, str] = {}
        self._suppressed_keys: Set[str] = set()
        self._suppression_counts: Dict[str, int] = {}
        self._is_finalized: bool = False
        self._total_targets_indexed: int = 0

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    @property
    def suppressed_keys(self) -> Set[str]:
        return set(self._suppressed_keys)

    @property
    def total_targets_indexed(self) -> int:
        return self._total_targets_indexed

    def add_target(
        self,
        target_id: str,
        source: str,
        keys: Iterable[str],
        country: Optional[str] = None,
    ) -> None:
        """Adds a target entity and its generated blocking keys to the index.

        Args:
            target_id: Target entity identifier (e.g. 's2_101', 's3_101').
            source: Originating target source ('Source 2' or 'Source 3').
            keys: Iterable of blocking key strings.
            country: Optional normalized country string.
        """
        if self._is_finalized:
            raise RuntimeError("Cannot add targets to an index that has already been finalized.")

        clean_id = str(target_id).strip()
        if not clean_id:
            return

        self._total_targets_indexed += 1
        self._target_sources[clean_id] = source.strip()
        if country is not None:
            self._target_countries[clean_id] = normalize_country(str(country))

        # Add to posting lists (avoid duplicates per key for the same entity)
        seen_keys: Set[str] = set()
        for k in keys:
            clean_k = str(k).strip()
            if clean_k and clean_k not in seen_keys:
                seen_keys.add(clean_k)
                self._posting_lists[clean_k].append(clean_id)

    def finalize(self, max_posting_list_size: int = 500) -> Dict[str, Any]:
        """Finalizes the index by pruning posting lists that violate the frequency guard.

        Args:
            max_posting_list_size: Maximum allowed postings per block key. Keys exceeding
                this size are recorded as suppressed and their posting lists cleared.

        Returns:
            Dictionary containing index finalization statistics.
        """
        if self._is_finalized:
            return self.get_stats()

        keys_to_suppress: List[Tuple[str, int]] = []
        max_size_observed = 0

        for key, postings in self._posting_lists.items():
            size = len(postings)
            if size > max_size_observed:
                max_size_observed = size
            if size > max_posting_list_size:
                keys_to_suppress.append((key, size))

        for key, size in keys_to_suppress:
            self._suppressed_keys.add(key)
            self._suppression_counts[key] = size
            del self._posting_lists[key]

        self._is_finalized = True

        stats = self.get_stats()
        stats["max_size_observed_before_suppression"] = max_size_observed
        return stats

    def query(self, keys: Iterable[str]) -> Tuple[Set[str], Dict[str, List[str]]]:
        """Queries the index with a collection of blocking keys.

        Args:
            keys: Iterable of blocking key strings for a Source 1 entity.

        Returns:
            Tuple of:
            - Set of matching candidate target IDs.
            - Dict mapping candidate target ID -> list of matched keys.
        """
        candidates: Set[str] = set()
        cand_to_keys: Dict[str, List[str]] = defaultdict(list)

        for k in keys:
            clean_k = str(k).strip()
            if not clean_k or clean_k in self._suppressed_keys:
                continue

            postings = self._posting_lists.get(clean_k)
            if postings:
                for target_id in postings:
                    candidates.add(target_id)
                    cand_to_keys[target_id].append(clean_k)

        return candidates, dict(cand_to_keys)

    def get_target_source(self, target_id: str) -> str:
        """Returns the source string ('Source 2' or 'Source 3') for a target ID."""
        if target_id in self._target_sources:
            return self._target_sources[target_id]
        if target_id.startswith("s2_") or target_id.startswith("S2_"):
            return "Source 2"
        if target_id.startswith("s3_") or target_id.startswith("S3_"):
            return "Source 3"
        return "Unknown"

    def get_target_country(self, target_id: str) -> str:
        """Returns the normalized country string for a target ID, or '' if unknown."""
        return self._target_countries.get(target_id, "")

    def get_stats(self) -> Dict[str, Any]:
        """Returns comprehensive index diagnostics and statistics."""
        active_keys_count = len(self._posting_lists)
        total_postings = sum(len(p) for p in self._posting_lists.values())
        avg_postings = (total_postings / active_keys_count) if active_keys_count > 0 else 0.0

        top_suppressed = sorted(
            self._suppression_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "index_name": self.name,
            "total_targets_indexed": self._total_targets_indexed,
            "active_blocking_keys": active_keys_count,
            "suppressed_keys_count": len(self._suppressed_keys),
            "total_postings_retained": total_postings,
            "avg_postings_per_active_key": round(avg_postings, 2),
            "top_suppressed_sample": top_suppressed,
            "is_finalized": self._is_finalized,
        }
