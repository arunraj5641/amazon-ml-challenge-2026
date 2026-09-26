"""Tests for standalone Phase 4 Blocking Recall Evaluator."""

import gzip
from pathlib import Path
import tempfile
import pytest

from src.blocking.recall_evaluator import (
    BlockingRecallEvaluator,
    load_authoritative_positives,
)


@pytest.fixture
def sample_ground_truth(tmp_path: Path) -> Path:
    """Creates a sample Phase 3 training pairs TSV."""
    gt_file = tmp_path / "training_pairs.tsv"
    content = (
        "source1_entity_id\ttarget_entity_id\ttarget_source\tlabel\tpair_type\n"
        "s1_001\ts2_001\tSource 2\t1\tpositive\n"
        "s1_001\ts3_001\tSource 3\t1\tpositive\n"
        "s1_002\ts2_002\tSource 2\t1\tpositive\n"
        "s1_002\ts2_999\tSource 2\t0\thard_negative\n"
        "s1_003\ts3_003\tSource 3\t1\tpositive\n"
        "s1_004\ts2_004\tSource 2\t1\tpositive\n"
        "s1_005\ts3_005\tSource 3\t1\tpositive\n"
    )
    gt_file.write_text(content, encoding="utf-8")
    return gt_file


@pytest.fixture
def sample_raw_gt(tmp_path: Path) -> Path:
    """Creates a sample raw train_ground_truth.tsv."""
    gt_file = tmp_path / "train_ground_truth.tsv"
    content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "s1_001\ts2_001,s3_001\n"
        "s1_002\ts2_002\n"
        "s1_003\t\n"
        "s1_004\ts2_004\n"
    )
    gt_file.write_text(content, encoding="utf-8")
    return gt_file


def test_load_authoritative_positives_training_pairs(sample_ground_truth: Path):
    positives = load_authoritative_positives(sample_ground_truth)
    assert len(positives) == 6
    # Verify deterministic sorting
    for i in range(len(positives) - 1):
        assert (positives[i][0], positives[i][1]) < (positives[i + 1][0], positives[i + 1][1])
    # Check negative pair was excluded
    assert not any(p[1] == "s2_999" for p in positives)


def test_load_authoritative_positives_raw_gt(sample_raw_gt: Path):
    positives = load_authoritative_positives(sample_raw_gt)
    assert len(positives) == 4
    expected = {
        ("s1_001", "s2_001", "Source 2"),
        ("s1_001", "s3_001", "Source 3"),
        ("s1_002", "s2_002", "Source 2"),
        ("s1_004", "s2_004", "Source 2"),
    }
    assert set(positives) == expected


def test_evaluator_all_positives_recovered(sample_ground_truth: Path, tmp_path: Path):
    positives = load_authoritative_positives(sample_ground_truth)
    # Generate candidates containing all positives plus extra negatives
    candidates = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_001\ts2_001\tSource 2\tcomposite\n",
        "s1_001\ts2_rand\tSource 2\tcomposite\n",
        "s1_001\ts3_001\tSource 3\tcomposite\n",
        "s1_002\ts2_002\tSource 2\tcomposite\n",
        "s1_003\ts3_003\tSource 3\tcomposite\n",
        "s1_004\ts2_004\tSource 2\tcomposite\n",
        "s1_005\ts3_005\tSource 3\tcomposite\n",
    ]
    missed_out = tmp_path / "missed.tsv.gz"
    evaluator = BlockingRecallEvaluator()
    report = evaluator.evaluate(
        candidate_lines=iter(candidates),
        sorted_ground_truth=positives,
        missed_output_tsv_path=missed_out,
    )

    assert report["overall"]["total_positive_pairs"] == 6
    assert report["overall"]["recovered_positive_pairs"] == 6
    assert report["overall"]["missed_positive_pairs"] == 0
    assert report["overall"]["blocking_recall"] == 1.0
    assert report["per_target_source"]["source2"]["recall"] == 1.0
    assert report["per_target_source"]["source3"]["recall"] == 1.0
    assert report["integrity_validation"]["status"] == "PASS"

    # Missed file should only contain the header
    with gzip.open(missed_out, "rt") as f:
        lines = f.readlines()
        assert len(lines) == 1


def test_evaluator_partial_recovery_and_missed_file(sample_ground_truth: Path, tmp_path: Path):
    positives = load_authoritative_positives(sample_ground_truth)
    # Recover only s1_001->s2_001 and s1_004->s2_004 (2 of 6)
    candidates = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_001\ts2_001\tSource 2\tcomposite\n",
        "s1_004\ts2_004\tSource 2\tcomposite\n",
    ]
    missed_out = tmp_path / "missed.tsv.gz"
    evaluator = BlockingRecallEvaluator()
    report = evaluator.evaluate(
        candidate_lines=iter(candidates),
        sorted_ground_truth=positives,
        missed_output_tsv_path=missed_out,
    )

    assert report["overall"]["total_positive_pairs"] == 6
    assert report["overall"]["recovered_positive_pairs"] == 2
    assert report["overall"]["missed_positive_pairs"] == 4
    assert report["overall"]["blocking_recall"] == round(2 / 6, 6)
    assert report["integrity_validation"]["status"] == "PASS"

    # Check missed file has exactly 4 missed pairs + 1 header
    with gzip.open(missed_out, "rt") as f:
        lines = f.readlines()
        assert len(lines) == 5
        missed_pairs = [line.strip().split("\t")[:2] for line in lines[1:]]
        assert ["s1_001", "s3_001"] in missed_pairs
        assert ["s1_002", "s2_002"] in missed_pairs
        assert ["s1_003", "s3_003"] in missed_pairs
        assert ["s1_005", "s3_005"] in missed_pairs


def test_evaluator_zero_recovery(sample_ground_truth: Path, tmp_path: Path):
    positives = load_authoritative_positives(sample_ground_truth)
    # Candidates with completely disjoint entities
    candidates = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_999\ts2_999\tSource 2\tcomposite\n",
    ]
    evaluator = BlockingRecallEvaluator()
    report = evaluator.evaluate(
        candidate_lines=iter(candidates),
        sorted_ground_truth=positives,
    )

    assert report["overall"]["total_positive_pairs"] == 6
    assert report["overall"]["recovered_positive_pairs"] == 0
    assert report["overall"]["missed_positive_pairs"] == 6
    assert report["overall"]["blocking_recall"] == 0.0
    assert report["s1_level_metrics"]["zero_recovery_s1_count"] == 5
    assert report["integrity_validation"]["status"] == "PASS"


def test_evaluator_duplicate_candidates_handling(sample_ground_truth: Path):
    positives = load_authoritative_positives(sample_ground_truth)
    # Candidate stream with duplicates
    candidates = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_001\ts2_001\tSource 2\tstrat1\n",
        "s1_001\ts2_001\tSource 2\tstrat2\n",  # duplicate candidate
        "s1_002\ts2_002\tSource 2\tstrat1\n",
    ]
    evaluator = BlockingRecallEvaluator()
    report = evaluator.evaluate(
        candidate_lines=iter(candidates),
        sorted_ground_truth=positives,
    )

    # s1_001->s2_001 should only be counted once
    assert report["overall"]["recovered_positive_pairs"] == 2
    assert report["overall"]["missed_positive_pairs"] == 4
    assert report["integrity_validation"]["status"] == "PASS"


def test_evaluator_s2_s3_breakdown(sample_ground_truth: Path):
    positives = load_authoritative_positives(sample_ground_truth)
    # Positives has 3 S2 (s1_001->s2_001, s1_002->s2_002, s1_004->s2_004)
    # and 3 S3 (s1_001->s3_001, s1_003->s3_003, s1_005->s3_005)
    candidates = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_001\ts2_001\tSource 2\tcomposite\n",
        "s1_002\ts2_002\tSource 2\tcomposite\n",
        # s1_004->s2_004 is missed -> S2 recall should be 2/3
        "s1_003\ts3_003\tSource 3\tcomposite\n",
        # s1_001->s3_001 and s1_005->s3_005 missed -> S3 recall should be 1/3
    ]
    evaluator = BlockingRecallEvaluator()
    report = evaluator.evaluate(
        candidate_lines=iter(candidates),
        sorted_ground_truth=positives,
    )

    s2 = report["per_target_source"]["source2"]
    s3 = report["per_target_source"]["source3"]
    assert s2["total_positives"] == 3
    assert s2["recovered"] == 2
    assert s2["missed"] == 1
    assert s2["recall"] == round(2 / 3, 6)

    assert s3["total_positives"] == 3
    assert s3["recovered"] == 1
    assert s3["missed"] == 2
    assert s3["recall"] == round(1 / 3, 6)
