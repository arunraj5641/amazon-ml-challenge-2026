"""I/O and persistence utilities for Phase 4 candidate pairs and evaluation artifacts."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.blocking.config import BlockingConfig
from src.blocking.types import CandidatePair, ValidationReport
from src.data.data_source import DataSource

logger = logging.getLogger(__name__)


def save_candidate_pairs(
    candidate_pairs: List[CandidatePair],
    output_source: DataSource,
    rel_path: str = "candidate_pairs.tsv",
) -> str:
    """Persists candidate pairs as a TSV table.

    Schema:
    source1_entity_id    target_entity_id    target_source    strategies
    """
    rows = []
    for p in candidate_pairs:
        rows.append(f"{p.source1_entity_id}\t{p.target_entity_id}\t{p.target_source}\t{','.join(p.strategies)}")

    content = "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n" + "\n".join(rows) + "\n"
    return output_source.write_text(rel_path, content)


def save_json_artifact(
    data: Dict[str, Any],
    output_source: DataSource,
    rel_path: str,
) -> str:
    """Persists JSON artifact cleanly with indentation and ISO timestamp support."""
    content = json.dumps(data, indent=2, default=str)
    return output_source.write_text(rel_path, content)


def save_all_blocking_artifacts(
    output_source: DataSource,
    candidate_pairs: List[CandidatePair],
    blocking_stats: Dict[str, Any],
    strategy_results: Dict[str, Any],
    validation_report: ValidationReport,
    config: BlockingConfig,
    git_commit: str = "unknown",
    normalization_version: str = "v001",
) -> Dict[str, str]:
    """Saves all Phase 4 artifacts to the configured output destination.

    Artifacts saved:
    - candidate_pairs.tsv
    - blocking_stats.json
    - blocking_strategy_results.json
    - blocking_metadata.json
    - blocking_validation_report.json

    Returns:
        Dict mapping artifact keys to their destination URIs/paths.
    """
    saved_uris: Dict[str, str] = {}

    # 1. candidate_pairs.tsv
    uri_cands = save_candidate_pairs(
        candidate_pairs=candidate_pairs,
        output_source=output_source,
        rel_path="candidate_pairs.tsv",
    )
    saved_uris["candidate_pairs_tsv"] = uri_cands

    # 2. blocking_stats.json
    uri_stats = save_json_artifact(
        data=blocking_stats,
        output_source=output_source,
        rel_path="blocking_stats.json",
    )
    saved_uris["blocking_stats_json"] = uri_stats

    # 3. blocking_strategy_results.json
    uri_strats = save_json_artifact(
        data=strategy_results,
        output_source=output_source,
        rel_path="blocking_strategy_results.json",
    )
    saved_uris["blocking_strategy_results_json"] = uri_strats

    # 4. blocking_validation_report.json
    uri_val = save_json_artifact(
        data=validation_report.to_dict(),
        output_source=output_source,
        rel_path="blocking_validation_report.json",
    )
    saved_uris["blocking_validation_report_json"] = uri_val

    # 5. blocking_metadata.json
    metadata = {
        "phase": 4,
        "phase_name": "candidate_blocking",
        "blocking_version": config.version,
        "input_normalization_version": normalization_version,
        "git_commit": git_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_candidate_pairs": len(candidate_pairs),
        "blocking_recall": blocking_stats.get("blocking_recall"),
        "candidate_reduction_ratio": blocking_stats.get("candidate_reduction_ratio"),
        "config": config.to_dict(),
        "validation_status": validation_report.status,
    }
    uri_meta = save_json_artifact(
        data=metadata,
        output_source=output_source,
        rel_path="blocking_metadata.json",
    )
    saved_uris["blocking_metadata_json"] = uri_meta

    return saved_uris


def save_candidate_pairs_file(
    file_path: Any,
    output_source: DataSource,
    rel_path: str = "candidate_pairs.tsv",
) -> str:
    """Persists a locally generated candidate_pairs.tsv file to DataSource without full-file RAM buffering.

    Supports direct copying for LocalDataSource and stream-based multipart upload for S3DataSource.
    """
    import shutil
    from src.data.data_source import LocalDataSource, S3DataSource

    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local candidate pairs file not found: {path}")

    if isinstance(output_source, LocalDataSource):
        target = output_source.base_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(path), str(target))
        return str(target)
    elif isinstance(output_source, S3DataSource):
        key = output_source._resolve_key(rel_path)
        output_source.client.upload_file(str(path), output_source.bucket, key)
        return f"s3://{output_source.bucket}/{key}"
    else:
        with open(path, "r", encoding="utf-8") as f:
            return output_source.write_text(rel_path, f.read())


def save_all_blocking_artifacts_streaming(
    output_source: DataSource,
    candidate_pairs_file: Any,
    total_candidate_pairs: int,
    blocking_stats: Dict[str, Any],
    strategy_results: Dict[str, Any],
    validation_report: ValidationReport,
    config: BlockingConfig,
    git_commit: str = "unknown",
    normalization_version: str = "v001",
) -> Dict[str, str]:
    """Saves all Phase 4 artifacts using a pre-written candidate pairs file on disk."""
    saved_uris: Dict[str, str] = {}

    # 1. candidate_pairs.tsv
    uri_cands = save_candidate_pairs_file(
        file_path=candidate_pairs_file,
        output_source=output_source,
        rel_path="candidate_pairs.tsv",
    )
    saved_uris["candidate_pairs_tsv"] = uri_cands

    # 2. blocking_stats.json
    uri_stats = save_json_artifact(
        data=blocking_stats,
        output_source=output_source,
        rel_path="blocking_stats.json",
    )
    saved_uris["blocking_stats_json"] = uri_stats

    # 3. blocking_strategy_results.json
    uri_strats = save_json_artifact(
        data=strategy_results,
        output_source=output_source,
        rel_path="blocking_strategy_results.json",
    )
    saved_uris["blocking_strategy_results_json"] = uri_strats

    # 4. blocking_validation_report.json
    uri_val = save_json_artifact(
        data=validation_report.to_dict(),
        output_source=output_source,
        rel_path="blocking_validation_report.json",
    )
    saved_uris["blocking_validation_report_json"] = uri_val

    # 5. blocking_metadata.json
    metadata = {
        "phase": 4,
        "phase_name": "candidate_blocking",
        "blocking_version": config.version,
        "input_normalization_version": normalization_version,
        "git_commit": git_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_candidate_pairs": total_candidate_pairs,
        "blocking_recall": blocking_stats.get("blocking_recall"),
        "candidate_reduction_ratio": blocking_stats.get("candidate_reduction_ratio"),
        "config": config.to_dict(),
        "validation_status": validation_report.status,
    }
    uri_meta = save_json_artifact(
        data=metadata,
        output_source=output_source,
        rel_path="blocking_metadata.json",
    )
    saved_uris["blocking_metadata_json"] = uri_meta

    return saved_uris

