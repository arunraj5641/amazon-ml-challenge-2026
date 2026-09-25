"""Phase 2 Normalization package."""

from src.normalization.normalizer import (
    COUNTRY_ALIASES,
    RECOGNIZED_LEGAL_SUFFIXES,
    UNICODE_NORMALIZATION_FORM,
    extract_business_name_core,
    normalize_business_address,
    normalize_business_address_alnum,
    normalize_business_name,
    normalize_business_name_alnum,
    normalize_country,
)
from src.normalization.pipeline import (
    NORMALIZATION_VERSION,
    FINAL_COLUMN_ORDER,
    NORMALIZED_COLUMNS_ADDED,
    SOURCE_FILES_TO_NORMALIZE,
    NormalizationPipeline,
)

__all__ = [
    "COUNTRY_ALIASES",
    "RECOGNIZED_LEGAL_SUFFIXES",
    "UNICODE_NORMALIZATION_FORM",
    "extract_business_name_core",
    "normalize_business_address",
    "normalize_business_address_alnum",
    "normalize_business_name",
    "normalize_business_name_alnum",
    "normalize_country",
    "NORMALIZATION_VERSION",
    "FINAL_COLUMN_ORDER",
    "NORMALIZED_COLUMNS_ADDED",
    "SOURCE_FILES_TO_NORMALIZE",
    "NormalizationPipeline",
]
