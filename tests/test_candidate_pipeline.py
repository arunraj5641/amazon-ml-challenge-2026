"""Unit tests for Phase 5 Candidate Pipeline and Validator."""

from pathlib import Path
import tempfile
import pytest

from src.candidates.config import CandidatePipelineConfig
from src.candidates.pipeline import CandidatePipeline
from src.candidates.validator import CandidateValidator


@pytest.fixture
def sample_candidate_lines():
    """Returns valid sorted candidate pair lines with header."""
    return [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_001\ts2_001\tSource 2\tcomposite,exact\n",
        "s1_001\ts2_002\tSource 2\tcomposite\n",
        "s1_001\ts3_001\tSource 3\tcomposite\n",
        "s1_002\ts2_003\tSource 2\tcomposite\n",
        "s1_002\ts3_002\tSource 3\tcomposite\n",
        "s1_003\ts2_004\tSource 2\tcomposite\n",
    ]


def test_candidate_pipeline_clean_run(sample_candidate_lines, tmp_path: Path):
    out_file = tmp_path / "candidate_pairs.tsv"
    config = CandidatePipelineConfig(expected_candidate_count=6)
    pipeline = CandidatePipeline(config=config)

    stats, report = pipeline.finalize(
        input_lines=iter(sample_candidate_lines),
        output_file_path=out_file,
        git_commit="test_hash_123",
    )

    assert report.status == "PASS"
    assert stats["total_input_candidates"] == 6
    assert stats["total_output_candidates"] == 6
    assert stats["unique_s1_entities"] == 3
    assert stats["duplicate_candidates_removed"] == 0
    assert stats["source_distribution"]["source2_candidates"] == 4
    assert stats["source_distribution"]["source3_candidates"] == 2
    assert stats["strategy_contributions"]["composite"] == 6
    assert stats["strategy_contributions"]["exact"] == 1

    # Verify output file content
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 7  # 1 header + 6 data
    assert lines[0] == "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies"
    assert lines[1] == "s1_001\ts2_001\tSource 2\tcomposite,exact"


def test_candidate_pipeline_duplicate_handling(tmp_path: Path):
    lines = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_001\ts2_001\tSource 2\tcomposite\n",
        "s1_001\ts2_001\tSource 2\tcomposite\n",  # Duplicate
        "s1_002\ts2_002\tSource 2\tcomposite\n",
    ]
    out_file = tmp_path / "candidate_pairs.tsv"
    pipeline = CandidatePipeline()
    stats, report = pipeline.finalize(
        input_lines=iter(lines),
        output_file_path=out_file,
    )

    assert stats["total_input_candidates"] == 3
    assert stats["total_output_candidates"] == 2
    assert stats["duplicate_candidates_removed"] == 1
    assert report.checks["5_duplicate_pairs"] == "WARN"


def test_candidate_pipeline_ordering_inversion(tmp_path: Path):
    lines = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_002\ts2_002\tSource 2\tcomposite\n",
        "s1_001\ts2_001\tSource 2\tcomposite\n",  # Inversion: s1_001 after s1_002
    ]
    out_file = tmp_path / "candidate_pairs.tsv"
    pipeline = CandidatePipeline()
    stats, report = pipeline.finalize(
        input_lines=iter(lines),
        output_file_path=out_file,
    )

    assert report.status == "FAIL"
    assert "4_deterministic_ordering" in report.checks
    assert report.checks["4_deterministic_ordering"] == "FAIL"


def test_candidate_pipeline_test_leakage(tmp_path: Path):
    lines = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "s1_001\ts2_001\tSource 2\tcomposite\n",
        "s1_test_leak\ts2_002\tSource 2\tcomposite\n",
    ]
    out_file = tmp_path / "candidate_pairs.tsv"
    pipeline = CandidatePipeline()
    stats, report = pipeline.finalize(
        input_lines=iter(lines),
        output_file_path=out_file,
        test_entity_ids={"s1_test_leak"},
    )

    assert report.status == "FAIL"
    assert report.checks["6_test_leakage"] == "FAIL"
    assert stats["total_output_candidates"] == 1  # Leaked row was dropped from output


def test_candidate_pipeline_empty_input(tmp_path: Path):
    lines = []
    out_file = tmp_path / "candidate_pairs.tsv"
    pipeline = CandidatePipeline(config=CandidatePipelineConfig(expected_candidate_count=0))
    stats, report = pipeline.finalize(
        input_lines=iter(lines),
        output_file_path=out_file,
        git_commit="git_123",
    )

    assert stats["total_input_candidates"] == 0
    assert stats["total_output_candidates"] == 0
    assert report.status == "PASS"


def test_candidate_pipeline_malformed_and_invalid_ids(tmp_path: Path):
    lines = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        "\n",  # empty line
        "malformed_single_token\n",
        "invalid_s1\ts2_001\tSource 2\tcomposite\n",  # invalid S1
        "s1_001\tinvalid_target\tSource 2\tcomposite\n",  # invalid target
    ]
    out_file = tmp_path / "candidate_pairs.tsv"
    pipeline = CandidatePipeline()
    stats, report = pipeline.finalize(
        input_lines=iter(lines),
        output_file_path=out_file,
    )

    assert report.status == "FAIL"
    assert report.checks["1_s1_id_format"] == "FAIL"
    assert report.checks["2_target_id_format"] == "FAIL"
