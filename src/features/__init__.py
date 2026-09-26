"""Phase 6 Feature Engineering package."""

from src.features.config import FEATURE_NAMES, FEATURE_VERSION, FeatureConfig
from src.features.extractor import EntityPreprocessedRecord, PairFeatureExtractor
from src.features.pipeline import FeaturePipeline
from src.features.schema import FeatureDefinition, build_default_feature_schema
from src.features.validator import FeatureValidationReport, FeatureValidator

__all__ = [
    "FEATURE_VERSION",
    "FEATURE_NAMES",
    "FeatureConfig",
    "FeatureDefinition",
    "build_default_feature_schema",
    "EntityPreprocessedRecord",
    "PairFeatureExtractor",
    "FeaturePipeline",
    "FeatureValidator",
    "FeatureValidationReport",
]
