"""Configuration parameters for Phase 6 Feature Engineering."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

FEATURE_VERSION = "feature_v001"

FEATURE_NAMES: List[str] = [
    # 1. Name features
    "name_exact_match",
    "name_core_exact_match",
    "name_alnum_exact_match",
    "name_char_length_diff",
    "name_char_length_ratio",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_token_dice",
    "name_token_containment",
    "name_prefix_match_3",
    "name_prefix_match_5",
    "name_char_qgram_jaccard",
    # 2. Address features
    "address_exact_match",
    "address_alnum_exact_match",
    "address_char_length_diff",
    "address_char_length_ratio",
    "address_token_jaccard",
    "address_token_overlap_count",
    "address_token_dice",
    "address_token_containment",
    "address_number_overlap_count",
    "address_number_jaccard",
    # 3. Country features
    "country_exact_match",
    "country_mismatch",
    "country_missing_either",
    "country_missing_both",
    # 4. Source & Strategy features
    "is_target_source2",
    "is_target_source3",
    "strategy_count",
    "strategy_exact_name",
    "strategy_name_token",
    "strategy_name_prefix",
    "strategy_address_conservative",
    "strategy_country_scoped_name",
    "strategy_composite",
    # 5. Missingness & Quality features
    "s1_name_missing",
    "target_name_missing",
    "s1_address_missing",
    "target_address_missing",
    "s1_country_missing",
    "target_country_missing",
    "both_address_missing",
    "total_available_fields",
    # 6. Entity length & token count features
    "s1_name_char_len",
    "target_name_char_len",
    "s1_name_token_count",
    "target_name_token_count",
    "s1_address_char_len",
    "target_address_char_len",
    "s1_address_token_count",
    "target_address_token_count",
]


@dataclass
class FeatureConfig:
    """Configuration for Phase 6 feature extraction pipeline."""

    version: str = FEATURE_VERSION
    chunksize: int = 50_000
    compress_output: bool = True
    max_rows: Optional[int] = None
    expected_train_pairs: int = 15_276_730
    expected_candidate_pairs: int = 499_912_583
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a serializable dictionary."""
        return asdict(self)
