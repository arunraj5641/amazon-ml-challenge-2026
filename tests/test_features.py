"""Unit tests for Phase 6 Feature Engineering pipeline and extractors."""

import gzip
from pathlib import Path
import tempfile
import pytest

from src.features.config import FEATURE_NAMES, FEATURE_VERSION, FeatureConfig
from src.features.extractor import EntityPreprocessedRecord, PairFeatureExtractor
from src.features.pipeline import FeaturePipeline
from src.features.schema import build_default_feature_schema
from src.features.validator import FeatureValidator


@pytest.fixture
def entity_acme_corp():
    return EntityPreprocessedRecord.from_row_dict({
        "entity_id": "s1_001",
        "business_name": "Acme Corporation Inc",
        "business_name_normalized": "acme corporation inc",
        "business_name_core": "acme",
        "business_name_alnum": "acmecorporationinc",
        "business_address": "123 Market St, Suite 400",
        "business_address_normalized": "123 market st ste 400",
        "business_address_alnum": "123marketstste400",
        "country": "US",
        "country_normalized": "us",
    })


@pytest.fixture
def entity_acme_ind():
    return EntityPreprocessedRecord.from_row_dict({
        "entity_id": "s2_001",
        "business_name": "Acme Industries LLC",
        "business_name_normalized": "acme industries llc",
        "business_name_core": "acme",
        "business_name_alnum": "acmeindustriesllc",
        "business_address": "123 Market St",
        "business_address_normalized": "123 market st",
        "business_address_alnum": "123marketst",
        "country": "US",
        "country_normalized": "us",
    })


@pytest.fixture
def entity_beta_empty():
    return EntityPreprocessedRecord.from_row_dict({
        "entity_id": "s3_999",
        "business_name": "",
        "business_name_normalized": "",
        "business_name_core": "",
        "business_name_alnum": "",
        "business_address": "",
        "business_address_normalized": "",
        "business_address_alnum": "",
        "country": "",
        "country_normalized": "",
    })


def test_schema_validity():
    schema = build_default_feature_schema()
    assert schema["version"] == FEATURE_VERSION
    assert schema["feature_count"] == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in schema["features"]
        feat = schema["features"][name]
        assert feat["name"] == name
        assert feat["dtype"] in ("float32", "int32")


def test_feature_extractor_identical_entities(entity_acme_corp):
    features = PairFeatureExtractor.extract_features(
        s1=entity_acme_corp,
        target=entity_acme_corp,
        target_source="Source 2",
        strategies_str="composite,exact_name",
    )
    assert len(features) == len(FEATURE_NAMES)
    idx_map = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # Name features
    assert features[idx_map["name_exact_match"]] == 1.0
    assert features[idx_map["name_core_exact_match"]] == 1.0
    assert features[idx_map["name_token_jaccard"]] == 1.0
    assert features[idx_map["name_char_length_diff"]] == 0.0
    assert features[idx_map["name_prefix_match_3"]] == 1.0

    # Address features
    assert features[idx_map["address_exact_match"]] == 1.0
    assert features[idx_map["address_token_jaccard"]] == 1.0

    # Country features
    assert features[idx_map["country_exact_match"]] == 1.0
    assert features[idx_map["country_mismatch"]] == 0.0

    # Strategy features
    assert features[idx_map["strategy_exact_name"]] == 1.0
    assert features[idx_map["strategy_composite"]] == 1.0


def test_feature_extractor_partial_match(entity_acme_corp, entity_acme_ind):
    features = PairFeatureExtractor.extract_features(
        s1=entity_acme_corp,
        target=entity_acme_ind,
        target_source="Source 2",
        strategies_str="composite,name_token",
    )
    idx_map = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # Exact name shouldn't match, but core name 'acme' should match
    assert features[idx_map["name_exact_match"]] == 0.0
    assert features[idx_map["name_core_exact_match"]] == 1.0
    assert features[idx_map["name_prefix_match_3"]] == 1.0  # both start with 'acm'
    assert features[idx_map["name_token_overlap_count"]] == 1.0  # 'acme' shared
    assert 0.0 < features[idx_map["name_token_jaccard"]] < 1.0

    # Address partial match (123 market st)
    assert features[idx_map["address_exact_match"]] == 0.0
    assert features[idx_map["address_token_containment"]] == 1.0  # target tokens fully in s1
    assert features[idx_map["address_number_overlap_count"]] == 1.0  # 123 shared

    # Strategy features
    assert features[idx_map["strategy_name_token"]] == 1.0
    assert features[idx_map["strategy_exact_name"]] == 0.0


def test_feature_extractor_empty_entity(entity_acme_corp, entity_beta_empty):
    features = PairFeatureExtractor.extract_features(
        s1=entity_acme_corp,
        target=entity_beta_empty,
        target_source="Source 3",
        strategies_str="",
    )
    idx_map = {name: i for i, name in enumerate(FEATURE_NAMES)}

    assert features[idx_map["name_exact_match"]] == 0.0
    assert features[idx_map["target_name_missing"]] == 1.0
    assert features[idx_map["target_address_missing"]] == 1.0
    assert features[idx_map["target_country_missing"]] == 1.0
    assert features[idx_map["s1_name_missing"]] == 0.0
    assert features[idx_map["is_target_source3"]] == 1.0
    assert features[idx_map["is_target_source2"]] == 0.0


def test_pipeline_training_and_candidate_compatibility(tmp_path: Path, entity_acme_corp, entity_acme_ind):
    pipeline = FeaturePipeline()
    pipeline.entities[entity_acme_corp.entity_id] = entity_acme_corp
    pipeline.entities[entity_acme_ind.entity_id] = entity_acme_ind

    # 1. Training pairs
    train_lines = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tlabel\tpair_type\n",
        f"{entity_acme_corp.entity_id}\t{entity_acme_ind.entity_id}\tSource 2\t1\tpositive\n",
    ]
    train_dir = tmp_path / "training"
    stats_tr, report_tr = pipeline.extract_training_features(iter(train_lines), output_dir=train_dir)
    assert report_tr.status == "PASS"

    # 2. Candidate pairs
    cand_lines = [
        "source1_entity_id\ttarget_entity_id\ttarget_source\tstrategies\n",
        f"{entity_acme_corp.entity_id}\t{entity_acme_ind.entity_id}\tSource 2\tcomposite\n",
    ]
    cand_dir = tmp_path / "candidates"
    stats_c, report_c = pipeline.extract_candidate_features(iter(cand_lines), output_dir=cand_dir)
    assert report_c.status == "PASS"

    # Verify column schemas match exactly
    with gzip.open(train_dir / "X_train.tsv.gz", "rt") as f_tr, \
         gzip.open(cand_dir / "X_candidates.tsv.gz", "rt") as f_c:
        header_tr = f_tr.readline().strip().split("\t")
        header_c = f_c.readline().strip().split("\t")
        assert header_tr == header_c
        assert header_tr == FEATURE_NAMES
        assert len(header_tr) == len(FEATURE_NAMES)

        row_tr = [float(x) for x in f_tr.readline().strip().split("\t")]
        row_c = [float(x) for x in f_c.readline().strip().split("\t")]
        assert len(row_tr) == len(FEATURE_NAMES)
        assert len(row_c) == len(FEATURE_NAMES)
        # Verify no NaN or Inf
        assert not any(x != x for x in row_tr)
        assert not any(x != x for x in row_c)

    # Verify training label was written separately
    with gzip.open(train_dir / "y_train.tsv.gz", "rt") as f_y:
        assert f_y.readline().strip() == "label"
        assert f_y.readline().strip() == "1"


def test_validator_detects_nan():
    validator = FeatureValidator()
    valid, nans, inv = validator.validate_features_batch(
        features=[[1.0, float("nan"), 3.0]],
        feature_names=["f1", "f2", "f3"],
    )
    assert nans == 1
    assert valid == 0
