"""Feature generation pipeline for Amazon ML Challenge 2026 Phase 6.

Transforms Phase 3 training pairs and Phase 5 final candidate pairs into
identical, normalized numerical feature matrices with O(1) pairwise RAM.
"""

from datetime import datetime, timezone
import gzip
import io
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import pandas as pd

from src.blocking.io import save_candidate_pairs_file, save_json_artifact
from src.blocking.recall_evaluator import stream_text_lines
from src.candidates.pipeline import get_git_commit_hash
from src.data.data_source import DataSource, LocalDataSource, S3DataSource, create_data_source
from src.features.config import FEATURE_NAMES, FeatureConfig
from src.features.extractor import EntityPreprocessedRecord, PairFeatureExtractor
from src.features.validator import FeatureValidationReport, FeatureValidator

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates streaming feature extraction for training and candidate datasets."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.config = config or FeatureConfig()
        self.logger = logger_instance or logger
        self.validator = FeatureValidator(logger_instance=self.logger)
        self.entities: Dict[str, EntityPreprocessedRecord] = {}

    def load_normalized_entities(
        self,
        source_paths: List[Union[str, Path]],
        region: Optional[str] = None,
    ) -> int:
        """Loads and precomputes compact entity records across normalized source tables.

        Args:
            source_paths: List of file paths or S3 URIs to normalized TSVs (S1, S2, S3).
            region: Optional AWS region.

        Returns:
            Total count of preprocessed entities loaded into memory cache.
        """
        self.logger.info("Loading normalized entities from %d source files...", len(source_paths))
        start_t = time.time()
        initial_count = len(self.entities)

        for src_path in source_paths:
            path_str = str(src_path).strip()
            self.logger.info("  Streaming entities from: %s", path_str)
            try:
                line_iter = stream_text_lines(path_str, region=region)
                first_line = next(line_iter, None)
            except Exception as e:
                self.logger.warning("Could not open entity file %s: %s", path_str, e)
                continue

            if not first_line:
                continue

            header = [h.strip() for h in first_line.rstrip("\r\n").split("\t")]
            col_map = {col: i for i, col in enumerate(header)}

            if "entity_id" not in col_map:
                self.logger.warning("File %s does not contain 'entity_id' column, skipping.", path_str)
                continue

            for line in line_iter:
                parts = line.rstrip("\r\n").split("\t")
                if not parts:
                    continue

                row_dict = {}
                for col, idx in col_map.items():
                    if idx < len(parts):
                        row_dict[col] = parts[idx].strip()

                e_id = row_dict.get("entity_id")
                if e_id:
                    self.entities[e_id] = EntityPreprocessedRecord.from_row_dict(row_dict)

        loaded_count = len(self.entities) - initial_count
        self.logger.info(
            "Successfully loaded %d new entities (total cached: %d) in %.1fs",
            loaded_count,
            len(self.entities),
            time.time() - start_t,
        )
        return len(self.entities)

    def extract_training_features(
        self,
        training_pairs_iter: Iterator[str],
        output_dir: Union[str, Path],
        git_commit: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], FeatureValidationReport]:
        """Streams Phase 3 training pairs and writes X_train, y_train, and pair_ids.

        Args:
            training_pairs_iter: Iterator yielding lines from training_pairs.tsv.
            output_dir: Local destination directory to write feature artifacts.
            git_commit: Git commit hash.
            max_rows: Optional row cap for testing / prototyping.

        Returns:
            Tuple of (stats_dict, validation_report).
        """
        self.logger.info("Starting Phase 6 feature generation for TRAINING pairs...")
        start_t = time.time()
        out_p = Path(output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        git_hash = git_commit or get_git_commit_hash()
        header_names = list(self.config.feature_names)

        x_path = out_p / "X_train.tsv.gz"
        y_path = out_p / "y_train.tsv.gz"
        ids_path = out_p / "pair_ids_train.tsv.gz"

        total_processed = 0
        missing_s1_entities = 0
        missing_target_entities = 0
        nan_inf_count = 0
        invalid_dim_count = 0

        # Feature running sums for statistics
        feature_sums = [0.0] * len(header_names)
        feature_mins = [float("inf")] * len(header_names)
        feature_maxs = [float("-inf")] * len(header_names)

        # Fallback dummy entity
        dummy_entity = EntityPreprocessedRecord.from_row_dict({})

        with gzip.open(x_path, "wt", encoding="utf-8") as f_x, \
             gzip.open(y_path, "wt", encoding="utf-8") as f_y, \
             gzip.open(ids_path, "wt", encoding="utf-8") as f_ids:

            # Write headers
            f_x.write("\t".join(header_names) + "\n")
            f_y.write("label\n")
            f_ids.write("source1_entity_id\ttarget_entity_id\ttarget_source\tpair_type\n")

            first_line = next(training_pairs_iter, None)
            if not first_line:
                val_report = self.validator.build_report(
                    dataset_name="training",
                    total_rows_processed=0,
                    feature_names=header_names,
                    nan_inf_count=0,
                    invalid_dim_count=0,
                    labels_present=True,
                    expected_rows=0,
                    git_commit=git_hash,
                )
                return {}, val_report

            header = [h.strip() for h in first_line.rstrip("\r\n").split("\t")]
            col_s1 = header.index("source1_entity_id") if "source1_entity_id" in header else 0
            col_target = header.index("target_entity_id") if "target_entity_id" in header else 1
            col_source = header.index("target_source") if "target_source" in header else -1
            col_label = header.index("label") if "label" in header else -1
            col_type = header.index("pair_type") if "pair_type" in header else -1

            batch_size = self.config.chunksize
            batch_lines_x = []
            batch_lines_y = []
            batch_lines_ids = []

            for line in training_pairs_iter:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(col_s1, col_target):
                    continue

                s1_id = parts[col_s1].strip()
                t_id = parts[col_target].strip()
                src = parts[col_source].strip() if col_source >= 0 and len(parts) > col_source else ""
                label_val = parts[col_label].strip() if col_label >= 0 and len(parts) > col_label else "0"
                pair_type = parts[col_type].strip() if col_type >= 0 and len(parts) > col_type else "unknown"

                s1_rec = self.entities.get(s1_id)
                if not s1_rec:
                    missing_s1_entities += 1
                    s1_rec = dummy_entity

                target_rec = self.entities.get(t_id)
                if not target_rec:
                    missing_target_entities += 1
                    target_rec = dummy_entity

                features = PairFeatureExtractor.extract_features(
                    s1=s1_rec,
                    target=target_rec,
                    target_source=src,
                    strategies_str="",
                )

                # Validation & stats
                if len(features) != len(header_names):
                    invalid_dim_count += 1
                else:
                    for i, val in enumerate(features):
                        feature_sums[i] += val
                        if val < feature_mins[i]:
                            feature_mins[i] = val
                        if val > feature_maxs[i]:
                            feature_maxs[i] = val

                batch_lines_x.append("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in features) + "\n")
                batch_lines_y.append(f"{label_val}\n")
                batch_lines_ids.append(f"{s1_id}\t{t_id}\t{src}\t{pair_type}\n")
                total_processed += 1

                if len(batch_lines_x) >= batch_size:
                    f_x.writelines(batch_lines_x)
                    f_y.writelines(batch_lines_y)
                    f_ids.writelines(batch_lines_ids)
                    batch_lines_x.clear()
                    batch_lines_y.clear()
                    batch_lines_ids.clear()

                if max_rows and total_processed >= max_rows:
                    break

            if batch_lines_x:
                f_x.writelines(batch_lines_x)
                f_y.writelines(batch_lines_y)
                f_ids.writelines(batch_lines_ids)

        elapsed = time.time() - start_t
        val_report = self.validator.build_report(
            dataset_name="training",
            total_rows_processed=total_processed,
            feature_names=header_names,
            nan_inf_count=nan_inf_count,
            invalid_dim_count=invalid_dim_count,
            labels_present=True,
            is_candidate_dataset=False,
            expected_rows=max_rows if max_rows else self.config.expected_train_pairs,
            git_commit=git_hash,
        )

        stats = {
            "dataset": "training",
            "total_pairs_processed": total_processed,
            "feature_count": len(header_names),
            "missing_s1_entity_lookups": missing_s1_entities,
            "missing_target_entity_lookups": missing_target_entities,
            "runtime_seconds": round(elapsed, 2),
            "rows_per_second": round(total_processed / elapsed, 1) if elapsed > 0 else 0.0,
            "feature_summaries": {
                name: {
                    "mean": round(feature_sums[i] / total_processed, 4) if total_processed > 0 else 0.0,
                    "min": feature_mins[i] if feature_mins[i] != float("inf") else 0.0,
                    "max": feature_maxs[i] if feature_maxs[i] != float("-inf") else 0.0,
                }
                for i, name in enumerate(header_names)
            },
        }

        self.logger.info(
            "Training feature extraction complete: %d pairs, %d features in %.1fs (%.0f pairs/sec)",
            total_processed,
            len(header_names),
            elapsed,
            stats["rows_per_second"],
        )

        return stats, val_report

    def extract_candidate_features(
        self,
        candidate_pairs_iter: Iterator[str],
        output_dir: Union[str, Path],
        git_commit: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], FeatureValidationReport]:
        """Streams Phase 5 final candidate pairs and writes X_candidates and candidate_ids.

        Args:
            candidate_pairs_iter: Iterator yielding lines from candidate_pairs.tsv.
            output_dir: Local destination directory to write candidate feature artifacts.
            git_commit: Git commit hash.
            max_rows: Optional row cap for testing / prototyping.

        Returns:
            Tuple of (stats_dict, validation_report).
        """
        self.logger.info("Starting Phase 6 feature generation for CANDIDATE pairs...")
        start_t = time.time()
        out_p = Path(output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        git_hash = git_commit or get_git_commit_hash()
        header_names = list(self.config.feature_names)

        x_path = out_p / "X_candidates.tsv.gz"
        ids_path = out_p / "candidate_ids.tsv.gz"

        total_processed = 0
        missing_s1_entities = 0
        missing_target_entities = 0
        nan_inf_count = 0
        invalid_dim_count = 0

        feature_sums = [0.0] * len(header_names)
        dummy_entity = EntityPreprocessedRecord.from_row_dict({})

        with gzip.open(x_path, "wt", encoding="utf-8") as f_x, \
             gzip.open(ids_path, "wt", encoding="utf-8") as f_ids:

            f_x.write("\t".join(header_names) + "\n")
            f_ids.write("source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n")

            first_line = next(candidate_pairs_iter, None)
            if not first_line:
                val_report = self.validator.build_report(
                    dataset_name="candidates",
                    total_rows_processed=0,
                    feature_names=header_names,
                    nan_inf_count=0,
                    invalid_dim_count=0,
                    labels_present=False,
                    is_candidate_dataset=True,
                    expected_rows=0,
                    git_commit=git_hash,
                )
                return {}, val_report

            header = [h.strip() for h in first_line.rstrip("\r\n").split("\t")]
            col_s1 = header.index("source1_entity_id") if "source1_entity_id" in header else 0
            col_target = header.index("target_entity_id") if "target_entity_id" in header else 1
            col_source = header.index("target_source") if "target_source" in header else 2
            col_strats = header.index("strategies") if "strategies" in header else 3

            batch_size = self.config.chunksize
            batch_lines_x = []
            batch_lines_ids = []

            for line in candidate_pairs_iter:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 2:
                    continue

                s1_id = parts[col_s1].strip()
                t_id = parts[col_target].strip()
                src = parts[col_source].strip() if len(parts) > col_source else ""
                strats = parts[col_strats].strip() if len(parts) > col_strats else ""

                s1_rec = self.entities.get(s1_id)
                if not s1_rec:
                    missing_s1_entities += 1
                    s1_rec = dummy_entity

                target_rec = self.entities.get(t_id)
                if not target_rec:
                    missing_target_entities += 1
                    target_rec = dummy_entity

                features = PairFeatureExtractor.extract_features(
                    s1=s1_rec,
                    target=target_rec,
                    target_source=src,
                    strategies_str=strats,
                )

                if len(features) != len(header_names):
                    invalid_dim_count += 1
                else:
                    for i, val in enumerate(features):
                        feature_sums[i] += val

                batch_lines_x.append("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in features) + "\n")
                batch_lines_ids.append(f"{s1_id}\t{t_id}\t{src}\t{strats}\n")
                total_processed += 1

                if len(batch_lines_x) >= batch_size:
                    f_x.writelines(batch_lines_x)
                    f_ids.writelines(batch_lines_ids)
                    batch_lines_x.clear()
                    batch_lines_ids.clear()

                if max_rows and total_processed >= max_rows:
                    break

            if batch_lines_x:
                f_x.writelines(batch_lines_x)
                f_ids.writelines(batch_lines_ids)

        elapsed = time.time() - start_t
        val_report = self.validator.build_report(
            dataset_name="candidates",
            total_rows_processed=total_processed,
            feature_names=header_names,
            nan_inf_count=nan_inf_count,
            invalid_dim_count=invalid_dim_count,
            labels_present=False,
            is_candidate_dataset=True,
            expected_rows=max_rows if max_rows else self.config.expected_candidate_pairs,
            git_commit=git_hash,
        )

        stats = {
            "dataset": "candidates",
            "total_pairs_processed": total_processed,
            "feature_count": len(header_names),
            "missing_s1_entity_lookups": missing_s1_entities,
            "missing_target_entity_lookups": missing_target_entities,
            "runtime_seconds": round(elapsed, 2),
            "rows_per_second": round(total_processed / elapsed, 1) if elapsed > 0 else 0.0,
            "feature_means": {
                name: round(feature_sums[i] / total_processed, 4) if total_processed > 0 else 0.0
                for i, name in enumerate(header_names)
            },
        }

        self.logger.info(
            "Candidate feature extraction complete: %d pairs, %d features in %.1fs (%.0f pairs/sec)",
            total_processed,
            len(header_names),
            elapsed,
            stats["rows_per_second"],
        )

        return stats, val_report
