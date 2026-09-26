"""Candidate generation engine orchestrating query lookup, deduplication, and country gating."""

import logging
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd

from src.blocking.config import BlockingConfig
from src.blocking.indexes import BlockingIndex
from src.blocking.strategies import BlockingStrategy, CompositeStrategy, get_field_or_derive
from src.blocking.types import CandidatePair
from src.data.data_source import DataSource

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Orchestrates candidate generation for Source 1 entities against a finalized BlockingIndex.

    Responsibilities:
    - Generates blocking keys per entity using configured strategy.
    - Queries target inverted index.
    - Deduplicates candidates across strategies.
    - Tracks which strategies matched each candidate.
    - Enforces country agreement policy without penalizing missing/unknown countries.
    - Guarantees deterministic candidate ordering.
    """

    def __init__(
        self,
        strategy: BlockingStrategy,
        index: BlockingIndex,
        config: Optional[BlockingConfig] = None,
    ):
        self.strategy = strategy
        self.index = index
        self.config = config or BlockingConfig()

    def generate_candidates_for_record(self, s1_record: Dict[str, Any]) -> List[CandidatePair]:
        """Generates candidate target pairs for a single Source 1 record.

        Args:
            s1_record: Dict representing a Source 1 entity row.

        Returns:
            Deterministically sorted list of CandidatePair instances.
        """
        s1_id = str(s1_record.get("entity_id", s1_record.get("source1_entity_id", ""))).strip()
        if not s1_id:
            return []

        s1_country = get_field_or_derive(s1_record, "country_normalized")

        # Map keys to generating strategy names
        if isinstance(self.strategy, CompositeStrategy):
            strat_key_map = self.strategy.generate_keys_with_strategy_map(s1_record)
            all_keys: List[str] = []
            key_to_strats: Dict[str, List[str]] = {}
            for s_name, keys in strat_key_map.items():
                for k in keys:
                    all_keys.append(k)
                    if k not in key_to_strats:
                        key_to_strats[k] = []
                    key_to_strats[k].append(s_name)
        else:
            all_keys = self.strategy.generate_keys(s1_record)
            key_to_strats = {k: [self.strategy.name] for k in all_keys}

        if not all_keys:
            return []

        # Deduplicate keys before query to avoid redundant posting list traversals
        unique_keys = list(dict.fromkeys(all_keys))

        # Query index
        candidate_ids, cand_matched_keys = self.index.query(unique_keys)
        if not candidate_ids:
            return []

        pairs: List[CandidatePair] = []
        for target_id in sorted(list(candidate_ids)):
            # Country agreement check
            if self.config.country_agreement_mode == "allow_missing":
                target_country = self.index.get_target_country(target_id)
                if s1_country and target_country and s1_country != target_country:
                    # Both specified but conflicting: skip candidate
                    continue
            elif self.config.country_agreement_mode == "strict":
                target_country = self.index.get_target_country(target_id)
                if not s1_country or not target_country or s1_country != target_country:
                    continue

            # Identify matching strategies
            matched_keys = cand_matched_keys.get(target_id, [])
            matching_strategies: Set[str] = set()
            for mk in matched_keys:
                for strat_name in key_to_strats.get(mk, []):
                    matching_strategies.add(strat_name)

            target_source = self.index.get_target_source(target_id)
            pair = CandidatePair(
                source1_entity_id=s1_id,
                target_entity_id=target_id,
                target_source=target_source,
                strategies=tuple(sorted(list(matching_strategies))),
            )
            pairs.append(pair)

        # Deterministic sort by target entity_id
        pairs.sort(key=lambda p: p.target_entity_id)
        return pairs

    def generate_candidates_batch(
        self, records: List[Dict[str, Any]]
    ) -> List[CandidatePair]:
        """Generates candidates for a batch of Source 1 records."""
        batch_pairs: List[CandidatePair] = []
        for rec in records:
            batch_pairs.extend(self.generate_candidates_for_record(rec))
        batch_pairs.sort(key=lambda p: (p.source1_entity_id, p.target_entity_id))
        return batch_pairs

    @classmethod
    def build_index_from_sources(
        cls,
        norm_source: DataSource,
        strategy: BlockingStrategy,
        config: Optional[BlockingConfig] = None,
        s2_rel_path: str = "train/train_source2_normalized.tsv",
        s3_rel_path: str = "train/train_source3_normalized.tsv",
        max_targets: Optional[int] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> BlockingIndex:
        """Streams target tables (Source 2 and Source 3) and constructs a finalized BlockingIndex.

        Args:
            norm_source: DataSource instance where normalized target tables reside.
            strategy: BlockingStrategy used to extract keys.
            config: BlockingConfig instance.
            s2_rel_path: Relative path to Source 2 table.
            s3_rel_path: Relative path to Source 3 table.
            max_targets: Optional target count cap for testing.
            logger_instance: Optional logger.

        Returns:
            Finalized BlockingIndex instance.
        """
        _log = logger_instance or logger
        cfg = config or BlockingConfig()
        index = BlockingIndex(name=f"index_{strategy.name}")

        def _resolve_path(path: str) -> str:
            if not norm_source.exists(path):
                alt = path.replace("_normalized.tsv", ".tsv")
                if norm_source.exists(alt):
                    return alt
            return path

        actual_s2 = _resolve_path(s2_rel_path)
        actual_s3 = _resolve_path(s3_rel_path)

        targets_loaded = 0

        # Stream Source 2
        _log.info("Indexing Source 2 targets from %s...", actual_s2)
        try:
            for chunk in norm_source.read_chunks(actual_s2, sep="\t", chunksize=cfg.chunksize):
                records = chunk.to_dict(orient="records")
                for rec in records:
                    e_id = str(rec.get("entity_id", "")).strip()
                    if not e_id:
                        continue
                    country = get_field_or_derive(rec, "country_normalized")
                    keys = strategy.generate_keys(rec)
                    index.add_target(
                        target_id=e_id,
                        source="Source 2",
                        keys=keys,
                        country=country,
                    )
                    targets_loaded += 1
                    if max_targets and targets_loaded >= max_targets:
                        break
                if max_targets and targets_loaded >= max_targets:
                    break
        except Exception as e:
            _log.warning("Could not read Source 2 from %s: %s", actual_s2, e)

        # Stream Source 3
        _log.info("Indexing Source 3 targets from %s...", actual_s3)
        try:
            for chunk in norm_source.read_chunks(actual_s3, sep="\t", chunksize=cfg.chunksize):
                records = chunk.to_dict(orient="records")
                for rec in records:
                    e_id = str(rec.get("entity_id", "")).strip()
                    if not e_id:
                        continue
                    country = get_field_or_derive(rec, "country_normalized")
                    keys = strategy.generate_keys(rec)
                    index.add_target(
                        target_id=e_id,
                        source="Source 3",
                        keys=keys,
                        country=country,
                    )
                    targets_loaded += 1
                    if max_targets and targets_loaded >= max_targets:
                        break
                if max_targets and targets_loaded >= max_targets:
                    break
        except Exception as e:
            _log.warning("Could not read Source 3 from %s: %s", actual_s3, e)

        _log.info(
            "Finalizing index for strategy '%s' (targets=%d, max_posting_size=%d)...",
            strategy.name,
            targets_loaded,
            cfg.max_posting_list_size,
        )
        final_stats = index.finalize(max_posting_list_size=cfg.max_posting_list_size)
        _log.info(
            "Index finalized: %d active keys, %d suppressed keys (postings retained: %d)",
            final_stats["active_blocking_keys"],
            final_stats["suppressed_keys_count"],
            final_stats["total_postings_retained"],
        )
        return index
