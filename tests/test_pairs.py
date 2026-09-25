"""Unit tests for Phase 3 Training Pair Builder."""

from pathlib import Path
import pytest
import pandas as pd

from src.data.data_source import LocalDataSource
from src.pairs.builder import TrainingPairBuilder, TrainingPairConfig
from src.pairs.profiler import GroundTruthProfiler
from src.pairs.sampler import NegativeSampler
from src.pairs.validator import PairValidator

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_data"


def test_ground_truth_profiler():
    """Verify ground-truth profiler extracts exact cardinalities and totals from sample data."""
    ds = LocalDataSource(base_dir=FIXTURES_DIR)
    profiler = GroundTruthProfiler(data_source=ds, ground_truth_rel_path="train/train_ground_truth.tsv")
    profile = profiler.profile()

    assert profile.total_source1_entities == 4
    assert profile.ground_truth_rows == 4
    assert profile.unique_source1_entities == 4
    assert profile.total_positive_pairs == 3
    assert profile.zero_match_count == 2
    assert profile.one_match_count == 1
    assert profile.multi_match_count == 1
    assert profile.cardinality_distribution[0] == 2
    assert profile.cardinality_distribution[1] == 1
    assert profile.cardinality_distribution[2] == 1
    assert profile.duplicate_references_within_row == 0
    assert profile.referenced_s2_entity_counts == 2
    assert profile.referenced_s3_entity_counts == 1
    assert profile.total_unique_targets_referenced == 3
    assert profile.invalid_references_count == 0
    assert profile.unexpected_entity_id_problems_count == 0

    p_dict = profile.to_dict()
    assert "cardinality_details" in p_dict
    assert p_dict["referenced_s2_entity_counts"] == 2


def test_negative_sampler_reproducibility_and_zero_collision():
    """Verify NegativeSampler produces identical samples given the same seed and 0 collisions."""
    s1_id = "S1-100"
    true_matches = {"S2-200", "S3-300"}
    hard_candidates = ["S2-200", "S2-201", "S2-202", "S3-300", "S3-301"]
    fallback = ["S2-901", "S2-902", "S3-903", "S3-904"]

    sampler1 = NegativeSampler(seed=42, hard_negative_ratio=0.5)
    negs1 = sampler1.sample_negatives_for_source1(
        s1_id=s1_id,
        num_negatives=2,
        true_matches=true_matches,
        hard_candidates=hard_candidates,
        fallback_candidates=fallback,
    )

    sampler2 = NegativeSampler(seed=42, hard_negative_ratio=0.5)
    negs2 = sampler2.sample_negatives_for_source1(
        s1_id=s1_id,
        num_negatives=2,
        true_matches=true_matches,
        hard_candidates=hard_candidates,
        fallback_candidates=fallback,
    )

    # 1. Deterministic reproducibility
    assert negs1 == negs2

    # 2. Zero collision with ground truth true matches
    for neg in negs1:
        assert neg not in true_matches


def test_negative_sampler_with_country_and_metadata():
    """Verify same-country hard negative sampling and metadata tracking."""
    sampler = NegativeSampler(seed=42, hard_negative_ratio=0.5)
    country_pool = {
        "US": ["s2_us1", "s2_us2", "s2_us3"],
        "India": ["s2_in1", "s2_in2"],
    }
    global_pool = ["s2_us1", "s2_us2", "s2_us3", "s2_in1", "s2_in2", "s3_se1"]
    true_matches = {"s2_us1"}

    sampled_tuples = sampler.sample_negatives_by_country(
        s1_id="s1_101",
        country="US",
        num_negatives=2,
        true_matches=true_matches,
        country_candidate_pool=country_pool,
        global_fallback_pool=global_pool,
    )

    assert len(sampled_tuples) == 2
    for target_id, p_type in sampled_tuples:
        assert target_id not in true_matches
        assert p_type in ("hard_negative", "random_negative")


def test_pair_validator():
    """Verify PairValidator detects duplicates, false negatives, and test leakage."""
    validator = PairValidator()
    authoritative_positives = {("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2")}
    test_ids = {"T1-100", "T2-200"}

    # Valid pairs
    clean_df = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-1", "label": 1},
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-99", "label": 0},
            {"source1_entity_id": "S1-2", "target_entity_id": "S2-2", "label": 1},
        ]
    )
    res_clean = validator.validate(clean_df, authoritative_positives=authoritative_positives, test_entity_ids=test_ids)
    assert res_clean.status == "PASS"
    assert res_clean.total_positive_pairs == 2
    assert res_clean.total_negative_pairs == 1
    assert res_clean.validation_checks["1_positive_fidelity"] == "PASS"
    assert res_clean.validation_checks["2_negative_integrity"] == "PASS"

    # Contaminated with false negative (a negative that is actually a true positive)
    bad_neg_df = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "target_entity_id": "S2-1", "label": 0},  # Error: true match labeled 0
        ]
    )
    res_bad = validator.validate(bad_neg_df, authoritative_positives=authoritative_positives)
    assert res_bad.status == "FAIL"
    assert any("Negative pair" in err for err in res_bad.errors)

    # Contaminated with test leakage
    leak_df = pd.DataFrame(
        [
            {"source1_entity_id": "T1-100", "target_entity_id": "S2-99", "label": 0},
        ]
    )
    res_leak = validator.validate(leak_df, authoritative_positives=authoritative_positives, test_entity_ids=test_ids)
    assert res_leak.status == "FAIL"
    assert res_leak.test_leakage_detected == 1


def test_pair_validator_strict_completeness_and_invalid_ids():
    """Verify PairValidator detects missing positives when require_all_positives is True."""
    validator = PairValidator()
    authoritative_positives = {("s1_1", "s2_1"), ("s1_1", "s3_1")}

    # Missing s1_1 -> s3_1
    partial_df = pd.DataFrame(
        [
            {"source1_entity_id": "s1_1", "target_entity_id": "s2_1", "label": 1},
            {"source1_entity_id": "s1_1", "target_entity_id": "s2_9", "label": 0},
        ]
    )
    res_partial = validator.validate(partial_df, authoritative_positives=authoritative_positives, require_all_positives=True)
    assert res_partial.status == "FAIL"
    assert res_partial.missing_authoritative_positives == 1
    assert res_partial.validation_checks["7_all_positives_preserved"] == "FAIL"

    # Invalid target ID prefix
    invalid_id_df = pd.DataFrame(
        [
            {"source1_entity_id": "s1_1", "target_entity_id": "unknown_99", "label": 0},
        ]
    )
    res_invalid = validator.validate(invalid_id_df, authoritative_positives=set())
    assert res_invalid.status == "FAIL"
    assert res_invalid.invalid_ids_detected >= 1


def test_training_pair_builder_end_to_end(tmp_path: Path):
    """Verify full end-to-end building and artifact persistence on sample data."""
    in_ds = LocalDataSource(base_dir=FIXTURES_DIR)
    out_ds = LocalDataSource(base_dir=tmp_path / "pairs")

    builder = TrainingPairBuilder(
        input_source=in_ds,
        output_source=out_ds,
        config=TrainingPairConfig(negatives_per_positive=1, seed=42),
    )

    candidate_pool_s2 = ["s2_101", "s2_102", "s2_105"]
    candidate_pool_s3 = ["s3_101", "s3_106"]

    pairs_df, stats = builder.build_pairs_from_ground_truth(
        gt_rel_path="train/train_ground_truth.tsv",
        candidate_pool_s2=candidate_pool_s2,
        candidate_pool_s3=candidate_pool_s3,
    )

    assert len(pairs_df) > 0
    assert stats["positive_pairs"] == 3
    assert stats["negative_pairs"] == 3
    assert stats["singleton_source1_entities"] == 2
    assert stats["multi_match_source1_entities"] == 1
    assert "pair_type" in pairs_df.columns

    # Validate generated pairs with require_all_positives=True
    authoritative_positives = {("s1_101", "s2_101"), ("s1_101", "s3_101"), ("s1_102", "s2_102")}
    val_res = builder.validator.validate(
        pairs_df,
        authoritative_positives=authoritative_positives,
        require_all_positives=True,
    )
    assert val_res.status == "PASS"
    assert val_res.all_positives_preserved is True
    assert val_res.multi_match_preserved is True

    # Save artifacts
    saved = builder.save_artifacts(pairs_df, stats, validation_result=val_res)
    assert Path(saved["training_pairs_tsv"]).is_file()
    assert out_ds.exists("pair_stats.json")
    assert out_ds.exists("pair_generation_report.json")
    assert out_ds.exists("pair_validation_report.json")
    assert out_ds.exists("metadata.json")


def test_builder_deterministic_reproducibility():
    """Verify builder produces identical pairs across separate runs with same seed."""
    in_ds = LocalDataSource(base_dir=FIXTURES_DIR)
    out_ds = LocalDataSource(base_dir=FIXTURES_DIR)

    builder1 = TrainingPairBuilder(
        input_source=in_ds,
        output_source=out_ds,
        config=TrainingPairConfig(negatives_per_positive=2, seed=123),
    )
    builder2 = TrainingPairBuilder(
        input_source=in_ds,
        output_source=out_ds,
        config=TrainingPairConfig(negatives_per_positive=2, seed=123),
    )

    cands = ["s2_101", "s2_102", "s2_105", "s3_101", "s3_106"]
    df1, s1 = builder1.build_pairs_from_ground_truth(
        candidate_pool_s2=cands,
        candidate_pool_s3=[],
    )
    df2, s2 = builder2.build_pairs_from_ground_truth(
        candidate_pool_s2=cands,
        candidate_pool_s3=[],
    )

    pd.testing.assert_frame_equal(df1, df2)
    assert s1["positive_pairs"] == s2["positive_pairs"]
    assert s1["negative_pairs"] == s2["negative_pairs"]


def test_candidate_pools_loader():
    """Verify TrainingPairBuilder.load_candidate_pools streams correctly."""
    ds = LocalDataSource(base_dir=FIXTURES_DIR)
    p_s2, p_s3, c_pools = TrainingPairBuilder.load_candidate_pools(
        norm_source=ds,
        s2_rel_path="train/train_source2.tsv",
        s3_rel_path="train/train_source3.tsv",
    )
    assert len(p_s2) == 3
    assert len(p_s3) == 2
    assert "US" in c_pools
