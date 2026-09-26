"""Evaluator for benchmarking modular blocking strategies against ground truth."""

import logging
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.blocking.config import BlockingConfig
from src.blocking.generator import CandidateGenerator
from src.blocking.indexes import BlockingIndex
from src.blocking.metrics import compute_blocking_metrics
from src.blocking.strategies import (
    BlockingStrategy,
    CompositeStrategy,
    create_default_strategies,
    create_strategy,
    get_field_or_derive,
)
from src.blocking.types import CandidatePair
from src.data.data_source import DataSource

logger = logging.getLogger(__name__)


class BlockingEvaluator:
    """Evaluates and benchmarks blocking strategies on training data against ground truth.

    Key capabilities:
    - Benchmarks individual strategy families in isolation.
    - Evaluates composite union strategies.
    - Accurately computes recall against authoritative ground truth.
    - Analyzes false negatives (missed matches) and suppression causes.
    - Supports deterministic sampling for rapid prototyping.
    """

    def __init__(
        self,
        norm_source: DataSource,
        raw_source: Optional[DataSource] = None,
        config: Optional[BlockingConfig] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.norm_source = norm_source
        self.raw_source = raw_source or norm_source
        self.config = config or BlockingConfig()
        self.logger = logger_instance or logger

    def load_ground_truth(
        self, gt_rel_path: str = "train/train_ground_truth.tsv"
    ) -> Tuple[Set[Tuple[str, str]], Dict[str, Set[str]], List[str]]:
        """Loads authoritative ground truth.

        Returns:
            Tuple of:
            - Set of (source1_entity_id, target_entity_id) true positive pairs.
            - Dict mapping source1_entity_id -> set of target_entity_ids.
            - Ordered list of all source1_entity_ids in ground truth.
        """
        self.logger.info("Loading ground truth from %s...", gt_rel_path)
        authoritative_positives: Set[Tuple[str, str]] = set()
        s1_matches_map: Dict[str, Set[str]] = {}
        all_s1_ids: List[str] = []

        actual_path = gt_rel_path
        if not self.raw_source.exists(actual_path):
            if self.norm_source.exists(actual_path):
                source = self.norm_source
            else:
                raise FileNotFoundError(f"Ground truth file not found: {actual_path}")
        else:
            source = self.raw_source

        for chunk in source.read_chunks(actual_path, sep="\t", chunksize=self.config.chunksize):
            for _, row in chunk.iterrows():
                s1_val = row.get("source1_entity_id")
                if s1_val is None or pd.isna(s1_val) or not str(s1_val).strip():
                    continue
                s1_id = str(s1_val).strip()
                all_s1_ids.append(s1_id)
                s1_matches_map[s1_id] = set()

                matches_val = row.get("matched_entity_ids")
                if (
                    matches_val is not None
                    and not pd.isna(matches_val)
                    and str(matches_val).strip()
                    and str(matches_val).strip().lower() not in ("nan", "null", "none", "<na>")
                ):
                    tokens = [t.strip() for t in str(matches_val).split(",") if t.strip()]
                    for target_id in tokens:
                        authoritative_positives.add((s1_id, target_id))
                        s1_matches_map[s1_id].add(target_id)

        self.logger.info(
            "Loaded ground truth: %d S1 entities, %d total positive pairs",
            len(all_s1_ids),
            len(authoritative_positives),
        )
        return authoritative_positives, s1_matches_map, all_s1_ids

    def load_source1_records(
        self,
        s1_rel_path: str = "train/train_source1_normalized.tsv",
        sampled_s1_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Loads Source 1 records, optionally filtering to a sample set."""
        actual_path = s1_rel_path
        if not self.norm_source.exists(actual_path):
            alt = actual_path.replace("_normalized.tsv", ".tsv")
            if self.norm_source.exists(alt):
                actual_path = alt
            elif self.raw_source.exists(alt):
                actual_path = alt

        self.logger.info("Loading Source 1 records from %s...", actual_path)
        records: List[Dict[str, Any]] = []
        for chunk in self.norm_source.read_chunks(actual_path, sep="\t", chunksize=self.config.chunksize):
            for rec in chunk.to_dict(orient="records"):
                e_id = str(rec.get("entity_id", rec.get("source1_entity_id", ""))).strip()
                if not e_id:
                    continue
                if sampled_s1_ids is None or e_id in sampled_s1_ids:
                    records.append(rec)

        self.logger.info("Loaded %d Source 1 records.", len(records))
        return records

    def evaluate_strategy(
        self,
        strategy: BlockingStrategy,
        s1_records: List[Dict[str, Any]],
        authoritative_positives: Set[Tuple[str, str]],
        s1_matches_map: Dict[str, Set[str]],
        total_target_pool_size: int,
        s2_rel_path: str = "train/train_source2_normalized.tsv",
        s3_rel_path: str = "train/train_source3_normalized.tsv",
        prebuilt_index: Optional[BlockingIndex] = None,
    ) -> Tuple[Dict[str, Any], List[CandidatePair], List[Dict[str, Any]]]:
        """Evaluates a single strategy (or composite) and measures metrics and false negatives.

        Returns:
            Tuple of:
            - Metrics dict.
            - Generated CandidatePair list.
            - False negatives analysis list.
        """
        start_time = time.time()
        self.logger.info("--- Evaluating Strategy: %s ---", strategy.name)

        # 1. Build or reuse index
        if prebuilt_index is not None:
            index = prebuilt_index
        else:
            index = CandidateGenerator.build_index_from_sources(
                norm_source=self.norm_source,
                strategy=strategy,
                config=self.config,
                s2_rel_path=s2_rel_path,
                s3_rel_path=s3_rel_path,
                logger_instance=self.logger,
            )

        # 2. Generate candidates
        generator = CandidateGenerator(strategy=strategy, index=index, config=self.config)
        candidate_pairs = generator.generate_candidates_batch(s1_records)
        runtime = time.time() - start_time

        # 3. Compute metrics
        evaluated_s1_ids = {
            str(r.get("entity_id", r.get("source1_entity_id", ""))).strip()
            for r in s1_records
            if str(r.get("entity_id", r.get("source1_entity_id", ""))).strip()
        }

        s1_country_map = {
            str(r.get("entity_id", r.get("source1_entity_id", ""))).strip(): get_field_or_derive(
                r, "country_normalized"
            )
            for r in s1_records
        }

        metrics = compute_blocking_metrics(
            candidate_pairs=candidate_pairs,
            authoritative_positives=authoritative_positives,
            evaluated_s1_ids=evaluated_s1_ids,
            total_target_pool_size=total_target_pool_size,
            s1_country_map=s1_country_map,
            s1_gt_matches_map=s1_matches_map,
            runtime_seconds=runtime,
            strategy_name=strategy.name,
        )

        metrics["index_stats"] = index.get_stats()

        # 4. Analyze false negatives (missed matches)
        generated_set = {(p.source1_entity_id, p.target_entity_id) for p in candidate_pairs}
        relevant_positives = {
            (s1, t) for (s1, t) in authoritative_positives if s1 in evaluated_s1_ids
        }
        missed = relevant_positives - generated_set

        fn_samples: List[Dict[str, Any]] = []
        for s1, t in sorted(list(missed))[:20]:
            fn_samples.append({
                "source1_id": s1,
                "target_id": t,
                "target_source": index.get_target_source(t),
                "s1_country": s1_country_map.get(s1, ""),
                "target_country": index.get_target_country(t),
            })

        self.logger.info(
            "Strategy '%s': Recall=%.4f, Candidates=%d, Reduction=%.4f (Runtime=%.2fs)",
            strategy.name,
            metrics["blocking_recall"],
            metrics["total_candidate_pairs"],
            metrics["candidate_reduction_ratio"],
            runtime,
        )

        return metrics, candidate_pairs, fn_samples

    def run_benchmark(
        self,
        strategy_names: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
        s1_rel_path: str = "train/train_source1_normalized.tsv",
        s2_rel_path: str = "train/train_source2_normalized.tsv",
        s3_rel_path: str = "train/train_source3_normalized.tsv",
        gt_rel_path: str = "train/train_ground_truth.tsv",
    ) -> Dict[str, Any]:
        """Runs comparative benchmark across multiple strategies and composite."""
        effective_seed = seed if seed is not None else self.config.seed
        effective_sample_size = sample_size if sample_size is not None else self.config.sample_size

        # 1. Load Ground Truth
        auth_positives, s1_matches_map, all_s1_ids = self.load_ground_truth(gt_rel_path)

        # 2. Apply deterministic sampling if requested
        if effective_sample_size and effective_sample_size < len(all_s1_ids):
            rng = random.Random(effective_seed)
            sampled_ids = set(rng.sample(all_s1_ids, effective_sample_size))
            self.logger.info(
                "Deterministic sampling active: %d / %d S1 entities selected (seed=%d)",
                len(sampled_ids),
                len(all_s1_ids),
                effective_seed,
            )
        else:
            sampled_ids = set(all_s1_ids)

        # 3. Load S1 records
        s1_records = self.load_source1_records(s1_rel_path, sampled_s1_ids=sampled_ids)

        # 4. Determine Target Pool Size
        available_strats = create_default_strategies(self.config)

        # Build target pool size count
        def _get_target_count(rel_p: str) -> int:
            actual = rel_p
            if not self.norm_source.exists(actual):
                alt = actual.replace("_normalized.tsv", ".tsv")
                if self.norm_source.exists(alt):
                    actual = alt
            cnt = 0
            try:
                for chunk in self.norm_source.read_chunks(actual, sep="\t", chunksize=self.config.chunksize):
                    cnt += len(chunk)
            except Exception:
                pass
            return cnt

        target_count = _get_target_count(s2_rel_path) + _get_target_count(s3_rel_path)
        self.logger.info("Total target pool size detected: %d", target_count)

        # 5. Evaluate requested strategies
        names_to_run = strategy_names or self.config.strategies
        if "composite" not in names_to_run:
            names_to_run = list(names_to_run) + ["composite"]

        benchmark_results: Dict[str, Any] = {}
        candidate_outputs: Dict[str, List[CandidatePair]] = {}
        false_negative_reports: Dict[str, List[Dict[str, Any]]] = {}

        for s_name in names_to_run:
            if s_name in available_strats:
                strat = available_strats[s_name]
            else:
                strat = create_strategy(s_name, self.config)

            metrics, cands, fn_analysis = self.evaluate_strategy(
                strategy=strat,
                s1_records=s1_records,
                authoritative_positives=auth_positives,
                s1_matches_map=s1_matches_map,
                total_target_pool_size=target_count,
                s2_rel_path=s2_rel_path,
                s3_rel_path=s3_rel_path,
            )
            benchmark_results[s_name] = metrics
            candidate_outputs[s_name] = cands
            false_negative_reports[s_name] = fn_analysis

        # Comparative summary table
        summary_comparison = []
        for s_name, res in benchmark_results.items():
            summary_comparison.append({
                "strategy": s_name,
                "recall": res["blocking_recall"],
                "candidate_pairs": res["total_candidate_pairs"],
                "reduction_ratio": res["candidate_reduction_ratio"],
                "avg_cands_per_s1": res["candidates_per_s1"]["average"],
                "p95_cands_per_s1": res["candidates_per_s1"]["p95"],
                "runtime_seconds": res["runtime_seconds"],
                "active_keys": res.get("index_stats", {}).get("active_blocking_keys", 0),
                "suppressed_keys": res.get("index_stats", {}).get("suppressed_keys_count", 0),
            })

        report = {
            "summary_comparison": summary_comparison,
            "strategy_details": benchmark_results,
            "false_negative_samples": false_negative_reports,
            "sample_size": effective_sample_size,
            "evaluated_s1_count": len(s1_records),
            "total_target_pool_size": target_count,
            "config": self.config.to_dict(),
        }

        # Keep candidate pairs for composite as primary candidate output
        best_candidates = candidate_outputs.get("composite", list(candidate_outputs.values())[0])

        return {
            "report": report,
            "candidates": best_candidates,
            "benchmark_results": benchmark_results,
        }
