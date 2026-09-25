"""Training pair builder orchestrating positive and negative pair construction for Phase 3.

Generates balanced, hard-negative enriched training pairs:
- Preserves every valid positive match from ground truth.
- Respects multi-match entities and singleton (zero-match) entities.
- Produces reproducible negative pairs using NegativeSampler.
- Validates pairs using PairValidator.
- Persists training_pairs.tsv, pair_stats.json, metadata.json, and validation/profiling reports.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd

from src.data.data_source import DataSource, LocalDataSource, S3DataSource
from src.pairs.profiler import GroundTruthProfile, GroundTruthProfiler
from src.pairs.sampler import NegativeSampler
from src.pairs.validator import PairValidationResult, PairValidator
from src.utils.logger import setup_logger

TRAINING_PAIRS_VERSION = "v001"


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "phase3-initial-commit"


@dataclass
class TrainingPairConfig:
    """Configuration parameters for Phase 3 pair generation."""

    negatives_per_positive: int = 1
    hard_negative_ratio: float = 0.5
    seed: int = 42
    chunksize: int = 100000
    version: str = TRAINING_PAIRS_VERSION
    raw_gt_prefix: str = "raw/dataset"
    normalized_prefix: str = "artifacts/normalized/v001"
    output_prefix: str = "artifacts/training_pairs/v001"


class TrainingPairBuilder:
    """Constructs, validates, and persists training pairs for the matcher."""

    def __init__(
        self,
        input_source: DataSource,
        output_source: DataSource,
        config: Optional[TrainingPairConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.input_source = input_source
        self.output_source = output_source
        self.config = config or TrainingPairConfig()
        self.logger = logger or setup_logger(name="pair_builder")
        self.sampler = NegativeSampler(
            seed=self.config.seed,
            hard_negative_ratio=self.config.hard_negative_ratio,
            logger=self.logger,
        )
        self.validator = PairValidator(logger=self.logger)

    @staticmethod
    def load_candidate_pools(
        norm_source: DataSource,
        s2_rel_path: str = "train/train_source2_normalized.tsv",
        s3_rel_path: str = "train/train_source3_normalized.tsv",
        chunksize: int = 100000,
        max_per_source: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """Streams normalized S2 and S3 sources to extract candidate target IDs and country pools.

        Returns:
            Tuple of (pool_s2, pool_s3, country_pools_dict).
        """
        _log = logger or logging.getLogger("load_candidate_pools")
        _log.info("Streaming candidate pools from %s and %s", s2_rel_path, s3_rel_path)

        pool_s2: List[str] = []
        pool_s3: List[str] = []
        country_pools: Dict[str, List[str]] = defaultdict(list)

        # Source 2
        s2_path = s2_rel_path
        if not norm_source.exists(s2_path):
            alt_s2 = s2_path.replace("_normalized.tsv", ".tsv")
            if norm_source.exists(alt_s2):
                s2_path = alt_s2

        try:
            s2_iter = norm_source.read_chunks(s2_path, sep="\t", chunksize=chunksize)
            for chunk in s2_iter:
                has_country = "country_normalized" in chunk.columns or "country" in chunk.columns
                country_col = "country_normalized" if "country_normalized" in chunk.columns else "country"

                for _, row in chunk.iterrows():
                    e_id = str(row["entity_id"]).strip()
                    pool_s2.append(e_id)
                    if has_country and not pd.isna(row.get(country_col)):
                        c_val = str(row[country_col]).strip()
                        if c_val:
                            country_pools[c_val].append(e_id)

                    if max_per_source and len(pool_s2) >= max_per_source:
                        break
                if max_per_source and len(pool_s2) >= max_per_source:
                    break
        except Exception as e:
            _log.warning("Could not stream candidate pool from %s: %s", s2_rel_path, e)

        # Source 3
        s3_path = s3_rel_path
        if not norm_source.exists(s3_path):
            alt_s3 = s3_path.replace("_normalized.tsv", ".tsv")
            if norm_source.exists(alt_s3):
                s3_path = alt_s3

        try:
            s3_iter = norm_source.read_chunks(s3_path, sep="\t", chunksize=chunksize)
            for chunk in s3_iter:
                has_country = "country_normalized" in chunk.columns or "country" in chunk.columns
                country_col = "country_normalized" if "country_normalized" in chunk.columns else "country"

                for _, row in chunk.iterrows():
                    e_id = str(row["entity_id"]).strip()
                    pool_s3.append(e_id)
                    if has_country and not pd.isna(row.get(country_col)):
                        c_val = str(row[country_col]).strip()
                        if c_val:
                            country_pools[c_val].append(e_id)

                    if max_per_source and len(pool_s3) >= max_per_source:
                        break
                if max_per_source and len(pool_s3) >= max_per_source:
                    break
        except Exception as e:
            _log.warning("Could not stream candidate pool from %s: %s", s3_rel_path, e)

        _log.info(
            "Extracted candidates: S2=%d, S3=%d, Distinct countries=%d",
            len(pool_s2),
            len(pool_s3),
            len(country_pools),
        )
        return pool_s2, pool_s3, dict(country_pools)

    def build_pairs_from_ground_truth(
        self,
        gt_rel_path: str = "train/train_ground_truth.tsv",
        candidate_pool_s2: Optional[List[str]] = None,
        candidate_pool_s3: Optional[List[str]] = None,
        country_candidate_pool: Optional[Dict[str, List[str]]] = None,
        s1_country_map: Optional[Dict[str, str]] = None,
        blocking_index: Optional[Dict[str, List[str]]] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Constructs positive and negative training pairs from ground truth.

        Args:
            gt_rel_path: Path to train_ground_truth.tsv.
            candidate_pool_s2: List of available target IDs in Source 2.
            candidate_pool_s3: List of available target IDs in Source 3.
            country_candidate_pool: Mapping of country -> list of candidate target IDs for hard negatives.
            s1_country_map: Mapping of source1_id -> country string.
            blocking_index: Optional mapping of blocking keys to target IDs for hard negatives.
            max_rows: Optional cap for testing/sample execution.

        Returns:
            Tuple of (DataFrame of training pairs, metadata report dict).
        """
        start_time = time.time()
        self.logger.info("Starting Training Pair Construction from: %s", self.input_source.get_uri(gt_rel_path))

        positive_pairs: List[Dict[str, Any]] = []
        negative_pairs: List[Dict[str, Any]] = []
        authoritative_positives: Set[Tuple[str, str]] = set()

        pool_s2 = candidate_pool_s2 or []
        pool_s3 = candidate_pool_s3 or []
        combined_pool = pool_s2 + pool_s3
        country_pool = country_candidate_pool or {}
        s1_countries = s1_country_map or {}
        b_index = blocking_index or {}

        # 1. Read Ground Truth and extract all authoritative positive pairs
        chunk_iter = self.input_source.read_chunks(
            rel_path=gt_rel_path,
            sep="\t",
            chunksize=self.config.chunksize,
        )

        rows_processed = 0
        singleton_s1_ids: List[str] = []
        multi_match_s1_ids: List[str] = []

        # Ground truth target pools (high-quality in-domain negatives)
        gt_target_pool: List[str] = []

        for chunk in chunk_iter:
            for _, row in chunk.iterrows():
                rows_processed += 1
                if max_rows and rows_processed > max_rows:
                    break

                raw_s1 = row.get("source1_entity_id")
                if raw_s1 is None or pd.isna(raw_s1) or not str(raw_s1).strip():
                    continue

                s1_id = str(raw_s1).strip()
                raw_matches = row.get("matched_entity_ids")

                if (
                    raw_matches is None
                    or pd.isna(raw_matches)
                    or not str(raw_matches).strip()
                    or str(raw_matches).strip().lower() in ("nan", "null", "none", "<na>")
                ):
                    singleton_s1_ids.append(s1_id)
                    continue

                tokens = [t.strip() for t in str(raw_matches).split(",") if t.strip()]
                if len(tokens) > 1:
                    multi_match_s1_ids.append(s1_id)

                for target_id in tokens:
                    t_lower = target_id.lower()
                    target_source = (
                        "Source 2"
                        if (t_lower.startswith("s2-") or t_lower.startswith("s2_") or "_s2_" in t_lower)
                        else "Source 3"
                    )
                    pair_tuple = (s1_id, target_id)
                    authoritative_positives.add(pair_tuple)
                    gt_target_pool.append(target_id)

                    positive_pairs.append(
                        {
                            "source1_entity_id": s1_id,
                            "target_entity_id": target_id,
                            "target_source": target_source,
                            "label": 1,
                            "pair_type": "positive",
                        }
                    )

            if max_rows and rows_processed > max_rows:
                break

        self.logger.info(
            "Extracted %d authoritative positive pairs across %d Ground Truth rows (%d singletons, %d multi-match).",
            len(positive_pairs),
            rows_processed,
            len(singleton_s1_ids),
            len(multi_match_s1_ids),
        )

        # Ensure we have a candidate pool for negative sampling
        effective_fallback_pool = combined_pool if combined_pool else list(set(gt_target_pool))

        # 2. Group positive target IDs per Source 1 entity for negative sampling
        positives_by_s1: Dict[str, Set[str]] = defaultdict(set)
        for s1, target in authoritative_positives:
            positives_by_s1[s1].add(target)

        # 3. Sample negative pairs for each matched Source 1 entity
        hard_neg_count = 0
        random_neg_count = 0

        if effective_fallback_pool:
            # Reset RNG to guarantee exact seed reproducibility
            self.sampler.reset_seed(self.config.seed)

            for s1_id, true_matches in positives_by_s1.items():
                num_needed = len(true_matches) * self.config.negatives_per_positive

                # Build hard candidate candidates:
                hard_candidates: List[str] = []
                # Lexical / blocking index candidates
                if s1_id in b_index:
                    hard_candidates.extend(b_index[s1_id])

                # Same-country candidates if available
                if s1_id in s1_countries:
                    c = s1_countries[s1_id]
                    if c in country_pool:
                        c_candidates = country_pool[c]
                        if len(c_candidates) > 50:
                            hard_candidates.extend(self.sampler.rng.sample(c_candidates, min(50, len(c_candidates))))
                        else:
                            hard_candidates.extend(c_candidates)

                sampled_tuples = self.sampler.sample_negatives_with_metadata(
                    s1_id=s1_id,
                    num_negatives=num_needed,
                    true_matches=true_matches,
                    hard_candidates=hard_candidates,
                    fallback_candidates=effective_fallback_pool,
                )

                for t_id, pair_type in sampled_tuples:
                    t_lower = t_id.lower()
                    t_source = (
                        "Source 2"
                        if (t_lower.startswith("s2-") or t_lower.startswith("s2_") or "_s2_" in t_lower)
                        else "Source 3"
                    )
                    if pair_type == "hard_negative":
                        hard_neg_count += 1
                    else:
                        random_neg_count += 1

                    negative_pairs.append(
                        {
                            "source1_entity_id": s1_id,
                            "target_entity_id": t_id,
                            "target_source": t_source,
                            "label": 0,
                            "pair_type": pair_type,
                        }
                    )

        total_pairs = positive_pairs + negative_pairs
        pairs_df = pd.DataFrame(total_pairs)

        if pairs_df.empty:
            pairs_df = pd.DataFrame(
                columns=["source1_entity_id", "target_entity_id", "target_source", "label", "pair_type"]
            )

        duration = round(time.time() - start_time, 2)

        stats = {
            "version": self.config.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "total_pairs": len(pairs_df),
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
            "hard_negative_pairs": hard_neg_count,
            "random_negative_pairs": random_neg_count,
            "singleton_source1_entities": len(singleton_s1_ids),
            "multi_match_source1_entities": len(multi_match_s1_ids),
            "unique_source1_in_pairs": pairs_df["source1_entity_id"].nunique()
            if not pairs_df.empty
            else 0,
            "unique_targets_in_pairs": pairs_df["target_entity_id"].nunique()
            if not pairs_df.empty
            else 0,
            "negatives_per_positive_ratio": self.config.negatives_per_positive,
            "hard_negative_ratio": self.config.hard_negative_ratio,
            "random_seed": self.config.seed,
        }

        return pairs_df, stats

    def save_artifacts(
        self,
        pairs_df: pd.DataFrame,
        stats: Dict[str, Any],
        validation_result: Optional[PairValidationResult] = None,
        ground_truth_profile: Optional[GroundTruthProfile] = None,
    ) -> Dict[str, str]:
        """Persists training pairs, stats, metadata, and validation reports to destination."""
        saved_uris: Dict[str, str] = {}
        git_commit = get_git_commit_hash()

        # 1. Save training_pairs.tsv
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pairs_df.to_csv(tmp_path, sep="\t", index=False)

        if isinstance(self.output_source, LocalDataSource):
            dest = self.output_source.base_dir / "training_pairs.tsv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(dest)
            saved_uris["training_pairs_tsv"] = str(dest)
        elif isinstance(self.output_source, S3DataSource):
            s3_key = self.output_source._resolve_key("training_pairs.tsv")
            self.output_source.client.upload_file(str(tmp_path), self.output_source.bucket, s3_key)
            if tmp_path.exists():
                tmp_path.unlink()
            saved_uris["training_pairs_tsv"] = f"s3://{self.output_source.bucket}/{s3_key}"

        # 2. Save pair_stats.json
        stats_json = json.dumps(stats, indent=2)
        saved_uris["pair_stats_json"] = self.output_source.write_text("pair_stats.json", stats_json)

        # 3. Save pair_validation_report.json
        if validation_result:
            val_json = json.dumps(validation_result.to_dict(), indent=2)
            saved_uris["pair_validation_report_json"] = self.output_source.write_text(
                "pair_validation_report.json", val_json
            )

        # 4. Save ground_truth_profile.json
        if ground_truth_profile:
            gt_profile_json = json.dumps(ground_truth_profile.to_dict(), indent=2)
            saved_uris["ground_truth_profile_json"] = self.output_source.write_text(
                "ground_truth_profile.json", gt_profile_json
            )

        # 5. Save metadata.json
        metadata = {
            "version": self.config.version,
            "random_seed": self.config.seed,
            "negative_sampling_ratio": self.config.negatives_per_positive,
            "hard_negative_ratio": self.config.hard_negative_ratio,
            "input_versions": {
                "ground_truth": f"s3://{self.output_source.bucket}/{self.config.raw_gt_prefix}/train/train_ground_truth.tsv"
                if isinstance(self.output_source, S3DataSource)
                else f"{self.config.raw_gt_prefix}/train/train_ground_truth.tsv",
                "normalized": f"s3://{self.output_source.bucket}/{self.config.normalized_prefix}"
                if isinstance(self.output_source, S3DataSource)
                else self.config.normalized_prefix,
            },
            "output_destination": self.output_source.get_uri(),
            "row_counts": {
                "total_pairs": len(pairs_df),
                "positive_pairs": stats.get("positive_pairs", 0),
                "negative_pairs": stats.get("negative_pairs", 0),
                "hard_negative_pairs": stats.get("hard_negative_pairs", 0),
                "random_negative_pairs": stats.get("random_negative_pairs", 0),
            },
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "code_commit_version": git_commit,
            "validation_status": validation_result.status if validation_result else "UNCHECKED",
        }
        meta_json = json.dumps(metadata, indent=2)
        saved_uris["metadata_json"] = self.output_source.write_text("metadata.json", meta_json)

        # 6. Save comprehensive pair_generation_report.json
        report = {
            "status": validation_result.status if validation_result else "GENERATED",
            "git_commit": git_commit,
            "config": asdict(self.config),
            "stats": stats,
            "validation": validation_result.to_dict() if validation_result else None,
            "ground_truth_profile": ground_truth_profile.to_dict() if ground_truth_profile else None,
            "metadata": metadata,
        }
        report_json = json.dumps(report, indent=2)
        saved_uris["pair_generation_report_json"] = self.output_source.write_text(
            "pair_generation_report.json", report_json
        )

        self.logger.info("Saved Phase 3 Training Pair Artifacts:")
        for k, uri in saved_uris.items():
            self.logger.info("  %s: %s", k, uri)

        return saved_uris
