import gzip
import heapq
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import pandas as pd

from src.blocking.config import BlockingConfig
from src.blocking.indexes import BlockingIndex
from src.blocking.strategies import BlockingStrategy, CompositeStrategy, get_field_or_derive
from src.blocking.types import CandidatePair
from src.data.data_source import DataSource

logger = logging.getLogger(__name__)


def write_candidates_chunk(
    pairs: List[CandidatePair],
    chunk_file: Union[str, Path],
) -> int:
    """Sorts candidate pairs and spills them to a disk chunk file (gzip supported).

    Args:
        pairs: List of CandidatePair objects to write.
        chunk_file: Destination file path (if ends with .gz, compresses with gzip).

    Returns:
        Number of candidate pairs written.
    """
    path = Path(chunk_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort chunk deterministically by (source1_entity_id, target_entity_id)
    pairs.sort(key=lambda p: (p.source1_entity_id, p.target_entity_id))

    lines = [
        f"{p.source1_entity_id}\t{p.target_entity_id}\t{p.target_source}\t{','.join(p.strategies)}\n"
        for p in pairs
    ]

    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as f:
            f.writelines(lines)
    else:
        with open(path, "wt", encoding="utf-8") as f:
            f.writelines(lines)

    return len(pairs)


def stream_merged_candidate_chunks(
    chunk_files: List[Union[str, Path]],
    output_tsv_path: Optional[Union[str, Path]] = None,
) -> Iterator[CandidatePair]:
    """Deterministically merges sorted chunk files using streaming K-way merge (O(1) RAM).

    Args:
        chunk_files: List of sorted chunk file paths (plain .tsv or .tsv.gz).
        output_tsv_path: Optional local destination path to write final candidate_pairs.tsv.

    Yields:
        CandidatePair instances in globally sorted order (source1_entity_id, target_entity_id).
    """
    valid_chunks = [Path(p).resolve() for p in chunk_files if Path(p).is_file()]
    if not valid_chunks:
        if output_tsv_path:
            out_p = Path(output_tsv_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "wt", encoding="utf-8") as out_f:
                out_f.write("source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n")
        return

    out_file_handle = None
    if output_tsv_path:
        out_p = Path(output_tsv_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_file_handle = open(out_p, "wt", encoding="utf-8")
        out_file_handle.write("source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n")

    try:
        with ExitStack() as stack:
            iterators = []
            for cp in valid_chunks:
                if str(cp).endswith(".gz"):
                    fh = stack.enter_context(gzip.open(cp, "rt", encoding="utf-8"))
                else:
                    fh = stack.enter_context(open(cp, "rt", encoding="utf-8"))
                iterators.append(fh)

            # Deterministic K-way merge ordered by (s1_id, target_id)
            merged = heapq.merge(*iterators, key=lambda line: line.split("\t", 2)[:2])

            for line in merged:
                if out_file_handle:
                    out_file_handle.write(line)

                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    s1 = parts[0]
                    t = parts[1]
                    src = parts[2]
                    strats = tuple(parts[3].split(",")) if len(parts) > 3 and parts[3] else ()
                    yield CandidatePair(
                        source1_entity_id=s1,
                        target_entity_id=t,
                        target_source=src,
                        strategies=strats,
                    )
    finally:
        if out_file_handle:
            out_file_handle.close()



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
