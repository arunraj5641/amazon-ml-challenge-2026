"""Unit tests for Phase 4 Blocking and Candidate Generation framework."""

from pathlib import Path
import tempfile
import pytest
import pandas as pd

from src.blocking.config import BlockingConfig
from src.blocking.evaluator import BlockingEvaluator
from src.blocking.generator import CandidateGenerator
from src.blocking.indexes import BlockingIndex
from src.blocking.io import save_all_blocking_artifacts, save_candidate_pairs
from src.blocking.metrics import compute_blocking_metrics
from src.blocking.strategies import (
    AddressConservativeStrategy,
    CompositeStrategy,
    CountryScopedNameStrategy,
    ExactNameStrategy,
    NamePrefixStrategy,
    NameTokenStrategy,
    create_default_strategies,
    create_strategy,
)
from src.blocking.types import CandidatePair
from src.blocking.validator import BlockingValidator
from src.data.data_source import LocalDataSource

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_data"


# ==============================================================================
# 1. Strategy Tests
# ==============================================================================


def test_exact_name_strategy():
    """Verify ExactNameStrategy extracts exact canonical core name keys."""
    strat = ExactNameStrategy()

    # Raw name with legal suffix
    rec = {"business_name": "Acme Electronics Corp"}
    keys = strat.generate_keys(rec)
    assert keys == ["exact:acme electronics"]

    # Pre-normalized fields
    rec2 = {"business_name_core": "apex logistics"}
    keys2 = strat.generate_keys(rec2)
    assert keys2 == ["exact:apex logistics"]

    # Empty name
    assert strat.generate_keys({"business_name": ""}) == []


def test_name_token_strategy_and_stopwords():
    """Verify NameTokenStrategy generates token keys and skips stopwords/short tokens."""
    strat = NameTokenStrategy(BlockingConfig(min_token_length=3))

    rec = {"business_name_core": "the acme and sons electronics"}
    keys = strat.generate_keys(rec)
    # 'the' and 'and' are stopwords, 'sons' and 'acme' and 'electronics' retained
    assert "tok:acme" in keys
    assert "tok:electronics" in keys
    assert "tok:sons" in keys
    assert "tok:the" not in keys
    assert "tok:and" not in keys


def test_name_prefix_strategy():
    """Verify NamePrefixStrategy generates character prefix keys for typos and stems."""
    strat = NamePrefixStrategy(BlockingConfig(prefix_length=4))

    rec = {"business_name_core": "international supply chain"}
    keys = strat.generate_keys(rec)
    assert "pre4:inte" in keys or "pre4:supp" in keys or "pre4:chai" in keys


def test_address_conservative_strategy_and_suppression():
    """Verify AddressConservativeStrategy filters common address stopwords and generates street+number and zip."""
    strat = AddressConservativeStrategy()

    # Address with street number and street name
    rec = {"business_address_alnum": "100 market street suite 200 94105"}
    keys = strat.generate_keys(rec)

    # Must contain number+name combination or zip
    assert "addr_num:100_market" in keys
    assert "addr_zip:94105" in keys
    # Pathological standalone stopwords like 'street', 'suite' must NOT exist as standalone keys
    assert "addr_tok:street" not in keys
    assert "addr_tok:suite" not in keys


def test_country_scoped_name_strategy_open_set():
    """Verify CountryScopedNameStrategy handles open-set countries and missing/unknown countries."""
    strat = CountryScopedNameStrategy()

    # Known / open-set country France
    rec_fr = {"business_name_core": "boulangerie paul", "country_normalized": "fr"}
    keys_fr = strat.generate_keys(rec_fr)
    assert "c_name:fr:boulangerie paul" in keys_fr
    assert "c_tok:fr:boulangerie" in keys_fr

    # Unseen / open-set country Norway
    rec_no = {"business_name_core": "nordic fjord", "country": "Norway"}
    keys_no = strat.generate_keys(rec_no)
    assert "c_name:norway:nordic fjord" in keys_no

    # Missing country fallback
    rec_unk = {"business_name_core": "global enterprise", "country_normalized": ""}
    keys_unk = strat.generate_keys(rec_unk)
    assert "c_name:unknown:global enterprise" in keys_unk


def test_composite_strategy_union():
    """Verify CompositeStrategy unions keys from multiple sub-strategies."""
    cfg = BlockingConfig()
    exact = ExactNameStrategy(cfg)
    token = NameTokenStrategy(cfg)
    composite = CompositeStrategy([exact, token])

    rec = {"business_name_core": "acme global"}
    keys = composite.generate_keys(rec)

    assert "exact:acme global" in keys
    assert "tok:acme" in keys
    assert "tok:global" in keys

    strat_map = composite.generate_keys_with_strategy_map(rec)
    assert "exact_name" in strat_map
    assert "name_token" in strat_map


# ==============================================================================
# 2. Index & Frequency Guard Tests
# ==============================================================================


def test_blocking_index_posting_and_frequency_suppression():
    """Verify BlockingIndex prunes high-frequency keys exceeding max_posting_list_size."""
    index = BlockingIndex(name="test_index")

    # Add 5 entities on common key 'tok:common'
    for i in range(5):
        index.add_target(
            target_id=f"s2_{i}",
            source="Source 2",
            keys=["tok:common", f"tok:unique_{i}"],
        )

    # Finalize with max_posting_list_size = 3
    stats = index.finalize(max_posting_list_size=3)

    assert stats["suppressed_keys_count"] == 1
    assert "tok:common" in index.suppressed_keys
    # Unique keys must remain active
    assert "tok:unique_0" not in index.suppressed_keys

    # Querying suppressed key returns empty
    cands, _ = index.query(["tok:common"])
    assert len(cands) == 0

    # Querying active key returns target
    cands_unique, _ = index.query(["tok:unique_0"])
    assert cands_unique == {"s2_0"}


def test_index_source_and_country_metadata():
    """Verify BlockingIndex tracks target sources and country metadata correctly."""
    index = BlockingIndex()
    index.add_target(target_id="s2_101", source="Source 2", keys=["k1"], country="us")
    index.add_target(target_id="s3_201", source="Source 3", keys=["k1"], country="in")
    index.finalize(max_posting_list_size=10)

    assert index.get_target_source("s2_101") == "Source 2"
    assert index.get_target_source("s3_201") == "Source 3"
    assert index.get_target_country("s2_101") == "us"
    assert index.get_target_country("s3_201") == "in"


# ==============================================================================
# 3. Candidate Generator Tests
# ==============================================================================


def test_candidate_generator_deduplication_and_ordering():
    """Verify CandidateGenerator deduplicates candidates and guarantees deterministic sorting."""
    index = BlockingIndex()
    index.add_target("s2_200", "Source 2", ["k1", "k2"])
    index.add_target("s2_100", "Source 2", ["k1"])
    index.finalize()

    strat = ExactNameStrategy()
    # Mock strategy to return both k1 and k2
    class MockStrategy(ExactNameStrategy):
        def generate_keys(self, record):
            return ["k1", "k2"]

    generator = CandidateGenerator(strategy=MockStrategy(), index=index)
    pairs = generator.generate_candidates_for_record({"entity_id": "s1_1"})

    # Must be deduplicated (2 targets total, not 3)
    assert len(pairs) == 2
    # Must be deterministically sorted by target_id: s2_100 before s2_200
    assert pairs[0].target_entity_id == "s2_100"
    assert pairs[1].target_entity_id == "s2_200"


def test_candidate_generator_country_agreement_modes():
    """Verify CandidateGenerator country agreement modes: allow_missing vs strict vs none."""
    index = BlockingIndex()
    index.add_target("s2_us", "Source 2", ["shared_key"], country="us")
    index.add_target("s2_in", "Source 2", ["shared_key"], country="in")
    index.add_target("s2_unk", "Source 2", ["shared_key"], country="")
    index.finalize()

    class MockStrategy(ExactNameStrategy):
        def generate_keys(self, record):
            return ["shared_key"]

    # 1. allow_missing: S1 with 'us' should match 'us' and 'unk', but NOT 'in'
    gen_allow = CandidateGenerator(
        MockStrategy(), index, BlockingConfig(country_agreement_mode="allow_missing")
    )
    pairs_allow = gen_allow.generate_candidates_for_record(
        {"entity_id": "s1_1", "country_normalized": "us"}
    )
    target_ids_allow = {p.target_entity_id for p in pairs_allow}
    assert "s2_us" in target_ids_allow
    assert "s2_unk" in target_ids_allow
    assert "s2_in" not in target_ids_allow

    # 2. strict: S1 with 'us' should match ONLY 'us'
    gen_strict = CandidateGenerator(
        MockStrategy(), index, BlockingConfig(country_agreement_mode="strict")
    )
    pairs_strict = gen_strict.generate_candidates_for_record(
        {"entity_id": "s1_1", "country_normalized": "us"}
    )
    target_ids_strict = {p.target_entity_id for p in pairs_strict}
    assert target_ids_strict == {"s2_us"}

    # 3. none: matches all
    gen_none = CandidateGenerator(
        MockStrategy(), index, BlockingConfig(country_agreement_mode="none")
    )
    pairs_none = gen_none.generate_candidates_for_record(
        {"entity_id": "s1_1", "country_normalized": "us"}
    )
    target_ids_none = {p.target_entity_id for p in pairs_none}
    assert target_ids_none == {"s2_us", "s2_in", "s2_unk"}


# ==============================================================================
# 4. Metrics & Reduction Calculation Tests
# ==============================================================================


def test_compute_blocking_metrics_recall_and_reduction():
    """Verify recall, candidate reduction ratio, and percentiles."""
    cands = [
        CandidatePair("s1_1", "s2_101", "Source 2"),
        CandidatePair("s1_1", "s3_101", "Source 3"),
        CandidatePair("s1_2", "s2_102", "Source 2"),
        CandidatePair("s1_3", "s2_999", "Source 2"),  # False positive candidate
    ]
    auth_positives = {("s1_1", "s2_101"), ("s1_1", "s3_101"), ("s1_2", "s2_102")}
    evaluated_s1 = {"s1_1", "s1_2", "s1_3"}
    total_target_pool = 1000

    metrics = compute_blocking_metrics(
        candidate_pairs=cands,
        authoritative_positives=auth_positives,
        evaluated_s1_ids=evaluated_s1,
        total_target_pool_size=total_target_pool,
    )

    # 3 true positives, all 3 recovered -> 100% recall
    assert metrics["blocking_recall"] == 1.0
    assert metrics["recovered_true_positives"] == 3
    assert metrics["missed_true_positives"] == 0

    # Total Cartesian: 3 * 1000 = 3000. Total candidate pairs: 4.
    # Reduction ratio: 1 - 4/3000 = 0.998667
    assert metrics["total_candidate_pairs"] == 4
    assert metrics["candidate_reduction_ratio"] > 0.998


# ==============================================================================
# 5. Validator Tests
# ==============================================================================


def test_validator_rules_and_leakage():
    """Verify BlockingValidator enforces rules 1-7 (ID format, duplicates, test leakage)."""
    validator = BlockingValidator()

    # 1. Clean valid pairs
    clean_pairs = [
        CandidatePair("s1_1", "s2_101", "Source 2"),
        CandidatePair("s1_1", "s3_101", "Source 3"),
        CandidatePair("s1_2", "s2_102", "Source 2"),
    ]
    rep_clean = validator.validate(clean_pairs, config_dict={"seed": 42}, git_commit="abc")
    assert rep_clean.status == "PASS"

    # 2. Duplicate pairs
    dupe_pairs = [
        CandidatePair("s1_1", "s2_101", "Source 2"),
        CandidatePair("s1_1", "s2_101", "Source 2"),
    ]
    rep_dupe = validator.validate(dupe_pairs)
    assert rep_dupe.status == "FAIL"
    assert rep_dupe.checks["4_no_duplicate_pairs"] == "FAIL"

    # 3. Invalid target ID format
    bad_id_pairs = [
        CandidatePair("s1_1", "invalid_id_format", "Source 2"),
    ]
    rep_bad_id = validator.validate(bad_id_pairs)
    assert rep_bad_id.status == "FAIL"
    assert rep_bad_id.checks["2_target_id_format"] == "FAIL"

    # 4. Cross-source target ID (target is s1_...)
    cross_pairs = [
        CandidatePair("s1_1", "s1_99", "Source 2"),
    ]
    rep_cross = validator.validate(cross_pairs)
    assert rep_cross.status == "FAIL"
    assert rep_cross.checks["3_no_cross_source_ids"] == "FAIL"

    # 5. Test leakage
    test_leak_pairs = [
        CandidatePair("s1_1", "s2_101", "Source 2"),
        CandidatePair("s1_1", "test_source2_100", "Source 2"),
    ]
    rep_leak = validator.validate(test_leak_pairs, test_entity_ids={"test_source2_100"})
    assert rep_leak.status == "FAIL"
    assert rep_leak.checks["7_no_test_leakage"] == "FAIL"


def test_validator_id_format_regression_hyphen_and_underscore():
    """Regression test: Verify validator accepts both hyphen (s1-XXXX) and underscore (s1_XXXX) formats."""
    validator = BlockingValidator()

    # 1. Hyphenated IDs (s1-XXXX, s2-XXXX, s3-XXXX)
    hyphen_pairs = [
        CandidatePair("s1-10001", "s2-20001", "Source 2"),
        CandidatePair("s1-10001", "s3-30001", "Source 3"),
        CandidatePair("s1-10002", "s2-20002", "Source 2"),
        CandidatePair("s1-10002", "s3-30002", "Source 3"),
    ]
    rep_hyphen = validator.validate(
        hyphen_pairs, input_normalization_version="v001", git_commit="abc", config_dict={"seed": 42}
    )
    assert rep_hyphen.status == "PASS"
    assert rep_hyphen.checks["1_s1_id_format"] == "PASS"
    assert rep_hyphen.checks["2_target_id_format"] == "PASS"
    assert rep_hyphen.checks["5_deterministic_order"] == "PASS"
    assert len(rep_hyphen.errors) == 0

    # 2. Underscore IDs (s1_XXXX, s2_XXXX, s3_XXXX)
    underscore_pairs = [
        CandidatePair("s1_10001", "s2_20001", "Source 2"),
        CandidatePair("s1_10001", "s3_30001", "Source 3"),
        CandidatePair("s1_10002", "s2_20002", "Source 2"),
        CandidatePair("s1_10002", "s3_30002", "Source 3"),
    ]
    rep_underscore = validator.validate(
        underscore_pairs, input_normalization_version="v001", git_commit="abc", config_dict={"seed": 42}
    )
    assert rep_underscore.status == "PASS"
    assert rep_underscore.checks["1_s1_id_format"] == "PASS"
    assert rep_underscore.checks["2_target_id_format"] == "PASS"
    assert rep_underscore.checks["5_deterministic_order"] == "PASS"
    assert len(rep_underscore.errors) == 0

    # 3. Invalid prefixes for S1 and targets
    invalid_s1 = [
        CandidatePair("e1_100", "s2-200", "Source 2"),
        CandidatePair("invalid_s1", "s3_300", "Source 3"),
        CandidatePair("s2-100", "s3-200", "Source 3"),
    ]
    rep_bad_s1 = validator.validate(invalid_s1)
    assert rep_bad_s1.status == "FAIL"
    assert rep_bad_s1.checks["1_s1_id_format"] == "FAIL"
    assert "invalid Source 1 entity IDs" in rep_bad_s1.errors[0]

    invalid_targets = [
        CandidatePair("s1-100", "invalid_prefix", "Source 2"),
        CandidatePair("s1-100", "s4-200", "Source 3"),
        CandidatePair("s1-100", "target_999", "Source 2"),
    ]
    rep_bad_t = validator.validate(invalid_targets)
    assert rep_bad_t.status == "FAIL"
    assert rep_bad_t.checks["2_target_id_format"] == "FAIL"
    assert "invalid Target entity IDs" in rep_bad_t.errors[0]


def test_batch_generation_deterministic_global_ordering():
    """Verify CandidateGenerator.generate_candidates_batch produces globally sorted candidate pairs."""
    from src.blocking.strategies import ExactNameStrategy

    cfg = BlockingConfig()
    strat = ExactNameStrategy(cfg)
    index = BlockingIndex(name="order_test_index")

    # Add targets
    index.add_target("s2-002", "Source 2", ["exact:alpha"])
    index.add_target("s2-001", "Source 2", ["exact:alpha"])
    index.add_target("s3-001", "Source 3", ["exact:beta"])
    index.finalize(max_posting_list_size=100)

    generator = CandidateGenerator(strat, index, cfg)

    # Provide S1 records out of order
    records = [
        {"entity_id": "s1-z", "business_name": "alpha"},
        {"entity_id": "s1-a", "business_name": "alpha"},
        {"entity_id": "s1-m", "business_name": "beta"},
    ]

    batch_pairs = generator.generate_candidates_batch(records)

    # Must be globally sorted by (source1_entity_id, target_entity_id)
    pair_tuples = [(p.source1_entity_id, p.target_entity_id) for p in batch_pairs]
    assert pair_tuples == sorted(pair_tuples)
    assert pair_tuples[0] == ("s1-a", "s2-001")
    assert pair_tuples[1] == ("s1-a", "s2-002")
    assert pair_tuples[2] == ("s1-m", "s3-001")
    assert pair_tuples[3] == ("s1-z", "s2-001")
    assert pair_tuples[4] == ("s1-z", "s2-002")

    # Validate passes Rule 5 with 0 ordering warnings
    validator = BlockingValidator()
    val_report = validator.validate(batch_pairs)
    assert val_report.status == "PASS"
    assert val_report.checks["5_deterministic_order"] == "PASS"
    assert not any("Rule 5" in w for w in val_report.warnings)


# ==============================================================================
# 6. Small Synthetic Dataset Tests (Full ER Modalities)
# ==============================================================================


def test_synthetic_dataset_all_modalities():
    """Verify blocking behavior on a comprehensive synthetic dataset.

    Contains:
    - Exact match
    - Name abbreviation
    - Punctuation variation
    - Word-order variation
    - Minor typo
    - Address variation
    - Open-set countries (France, Germany, India, US)
    - Zero-match S1 (singleton)
    - Multi-match S1
    - Distractor records
    """
    # Synthetic Source 1 Records
    s1_data = [
        # 1. Exact match (US)
        {"entity_id": "s1_101", "business_name": "Apex Electronics Corp", "business_address": "100 Market St, SF", "country": "US"},
        # 2. Word-order variation + multi-match to S2 and S3 (India)
        {"entity_id": "s1_102", "business_name": "Global Logistics India Pvt Ltd", "business_address": "12 MG Road, Bangalore", "country": "India"},
        # 3. Punctuation and suffix variation (France - open set)
        {"entity_id": "s1_103", "business_name": "Boutique & Cafe Parisienne S.A.", "business_address": "45 Rue de Rivoli, Paris", "country": "France"},
        # 4. Typo in target name (Germany - open set)
        {"entity_id": "s1_104", "business_name": "Munich Engineering Solutions GmbH", "business_address": "Maximilianstrasse 10, Munich", "country": "Germany"},
        # 5. Zero-match S1 (Singleton)
        {"entity_id": "s1_105", "business_name": "Unique Lone Star Ventures", "business_address": "1 Prairie Way, Austin", "country": "US"},
    ]

    # Synthetic Source 2 Targets
    s2_data = [
        # Matches s1_101 exactly
        {"entity_id": "s2_101", "business_name": "Apex Electronics", "business_address": "100 Market Street", "country": "US"},
        # Matches s1_102 (word order: Logistics Global)
        {"entity_id": "s2_102", "business_name": "Logistics Global India", "business_address": "MG Road No 12", "country": "IN"},
        # Matches s1_103 (punctuation expanded: Boutique and Cafe Parisienne)
        {"entity_id": "s2_103", "business_name": "Boutique and Cafe Parisienne", "business_address": "45 Rue de Rivoli", "country": "FR"},
        # Matches s1_104 (typo: 'Enginering' instead of 'Engineering')
        {"entity_id": "s2_104", "business_name": "Munich Enginering Solutions", "business_address": "Maximilianstrasse", "country": "DE"},
        # Distractor record
        {"entity_id": "s2_999", "business_name": "Totally Unrelated Pharmacy", "business_address": "99 Broadway", "country": "US"},
    ]

    # Synthetic Source 3 Targets
    s3_data = [
        # Second match for s1_102 (Multi-match: s1_102 matches both s2_102 and s3_102)
        {"entity_id": "s3_102", "business_name": "Global Logistics India Enterprise", "business_address": "12 MG Rd, Bengaluru", "country": "India"},
        # Distractor record
        {"entity_id": "s3_888", "business_name": "Random Bakery Tokyo", "business_address": "Shibuya 1-1", "country": "JP"},
    ]

    # Authoritative Ground Truth
    auth_gt = {
        ("s1_101", "s2_101"),
        ("s1_102", "s2_102"),
        ("s1_102", "s3_102"),  # Multi-match
        ("s1_103", "s2_103"),  # France
        ("s1_104", "s2_104"),  # Germany (typo)
    }

    # Setup Composite Strategy
    cfg = BlockingConfig(
        strategies=["exact_name", "name_token", "name_prefix", "address_conservative", "country_scoped_name"],
        prefix_length=4,
        max_posting_list_size=10,
        country_agreement_mode="allow_missing",
    )
    strats = create_default_strategies(cfg)
    composite = strats["composite"]

    # Index Targets
    index = BlockingIndex(name="synthetic_index")
    for rec in s2_data:
        keys = composite.generate_keys(rec)
        index.add_target(rec["entity_id"], "Source 2", keys, country=rec.get("country"))
    for rec in s3_data:
        keys = composite.generate_keys(rec)
        index.add_target(rec["entity_id"], "Source 3", keys, country=rec.get("country"))
    index.finalize(max_posting_list_size=10)

    # Generate Candidates
    generator = CandidateGenerator(composite, index, cfg)
    all_candidates = generator.generate_candidates_batch(s1_data)

    cand_set = {(c.source1_entity_id, c.target_entity_id) for c in all_candidates}

    # Assert 100% of Ground Truth matches are recovered
    for true_pair in auth_gt:
        assert true_pair in cand_set, f"Ground truth match {true_pair} was missed by composite blocking!"

    # Multi-match check: s1_102 must have candidates in BOTH s2 and s3
    s1_102_targets = {c.target_entity_id for c in all_candidates if c.source1_entity_id == "s1_102"}
    assert "s2_102" in s1_102_targets
    assert "s3_102" in s1_102_targets

    # France open-set check
    s1_103_targets = {c.target_entity_id for c in all_candidates if c.source1_entity_id == "s1_103"}
    assert "s2_103" in s1_103_targets

    # Typo recovery check (caught by prefix 'muni')
    s1_104_targets = {c.target_entity_id for c in all_candidates if c.source1_entity_id == "s1_104"}
    assert "s2_104" in s1_104_targets

    # Distractor check: distractor records should NOT be matched to everything
    s1_105_targets = {c.target_entity_id for c in all_candidates if c.source1_entity_id == "s1_105"}
    assert len(s1_105_targets) == 0, "Singleton s1_105 should produce 0 candidates"

    # Compute Metrics
    s1_ids = {r["entity_id"] for r in s1_data}
    metrics = compute_blocking_metrics(
        candidate_pairs=all_candidates,
        authoritative_positives=auth_gt,
        evaluated_s1_ids=s1_ids,
        total_target_pool_size=len(s2_data) + len(s3_data),
    )
    assert metrics["blocking_recall"] == 1.0
    assert metrics["recovered_true_positives"] == 5
    assert metrics["candidate_reduction_ratio"] > 0.5


# ==============================================================================
# 7. End-to-End Evaluator & Persistence on Fixtures
# ==============================================================================


def test_blocking_evaluator_end_to_end(tmp_path: Path):
    """Verify BlockingEvaluator benchmarks and saves all artifacts on sample fixtures."""
    ds = LocalDataSource(base_dir=FIXTURES_DIR)
    out_ds = LocalDataSource(base_dir=tmp_path / "block_eval")

    cfg = BlockingConfig(
        strategies=["exact_name", "name_token", "composite"],
        max_posting_list_size=50,
        seed=42,
    )

    evaluator = BlockingEvaluator(
        norm_source=ds,
        raw_source=ds,
        config=cfg,
    )

    benchmark_out = evaluator.run_benchmark(
        strategy_names=["exact_name", "name_token", "composite"],
        s1_rel_path="train/train_source1.tsv",
        s2_rel_path="train/train_source2.tsv",
        s3_rel_path="train/train_source3.tsv",
        gt_rel_path="train/train_ground_truth.tsv",
    )

    report = benchmark_out["report"]
    candidates = benchmark_out["candidates"]
    assert len(report["summary_comparison"]) == 3
    assert len(candidates) > 0

    # Validate
    validator = BlockingValidator()
    val_report = validator.validate(candidates, config_dict=cfg.to_dict())
    assert val_report.status == "PASS"

    # Save artifacts
    saved = save_all_blocking_artifacts(
        output_source=out_ds,
        candidate_pairs=candidates,
        blocking_stats=report["strategy_details"]["composite"],
        strategy_results=report,
        validation_report=val_report,
        config=cfg,
    )

    assert out_ds.exists("candidate_pairs.tsv")
    assert out_ds.exists("blocking_stats.json")
    assert out_ds.exists("blocking_strategy_results.json")
    assert out_ds.exists("blocking_metadata.json")
    assert out_ds.exists("blocking_validation_report.json")


def test_duplicate_composite_keys_deduplicated_and_produces_candidate_once():
    """Verify duplicate keys across composite sub-strategies produce candidate target only once and capture all strategies."""
    cfg = BlockingConfig()

    # Create two strategies that produce an overlapping key 'overlap_key'
    class MockStrategyA(ExactNameStrategy):
        @property
        def name(self) -> str:
            return "strat_a"

        def generate_keys(self, record):
            return ["overlap_key", "unique_a"]

    class MockStrategyB(ExactNameStrategy):
        @property
        def name(self) -> str:
            return "strat_b"

        def generate_keys(self, record):
            return ["overlap_key", "unique_b"]

    strat_a = MockStrategyA(cfg)
    strat_b = MockStrategyB(cfg)
    composite = CompositeStrategy([strat_a, strat_b])

    # Index target entity on 'overlap_key'
    index = BlockingIndex(name="overlap_test_index")
    index.add_target(target_id="s2-100", source="Source 2", keys=["overlap_key"])
    index.finalize(max_posting_list_size=50)

    generator = CandidateGenerator(strategy=composite, index=index, config=cfg)

    # Query with record that generates overlapping keys
    s1_rec = {"entity_id": "s1-100", "business_name": "Test Company"}
    pairs = generator.generate_candidates_for_record(s1_rec)

    # Candidate target s2-100 must appear EXACTLY ONCE
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.source1_entity_id == "s1-100"
    assert pair.target_entity_id == "s2-100"
    # Both strat_a and strat_b must be recorded in strategies
    assert "strat_a" in pair.strategies
    assert "strat_b" in pair.strategies


def test_generate_candidates_cli_deterministic_global_ordering(tmp_path: Path):
    """Verify generate_candidates CLI outputs candidate pairs strictly ordered by (s1, target) and passes Rule 5."""
    import subprocess
    import sys

    out_dir = tmp_path / "gen_cli_order_test"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "scripts" / "generate_candidates.py"),
        "--norm-input",
        str(FIXTURES_DIR),
        "--output",
        str(out_dir),
        "--strategy",
        "composite",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert (out_dir / "candidate_pairs.tsv").is_file()

    # Read pairs from TSV and verify strict sorting
    df = pd.read_csv(out_dir / "candidate_pairs.tsv", sep="\t")
    pairs = list(zip(df["source1_entity_id"], df["target_entity_id"]))
    assert pairs == sorted(pairs)
    assert len(pairs) == len(set(pairs))  # 0 duplicates

