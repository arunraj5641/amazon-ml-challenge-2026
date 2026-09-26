"""Configuration parameters for Phase 4 Blocking and Candidate Generation."""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Set

BLOCKING_VERSION = "block_v001"

# Standard address stopwords that cause pathological candidate explosion if used for blocking
DEFAULT_ADDRESS_STOPWORDS: Set[str] = {
    "st",
    "street",
    "rd",
    "road",
    "ave",
    "avenue",
    "blvd",
    "boulevard",
    "dr",
    "drive",
    "ln",
    "lane",
    "way",
    "ct",
    "court",
    "pl",
    "place",
    "sq",
    "square",
    "ste",
    "suite",
    "fl",
    "floor",
    "apt",
    "apartment",
    "unit",
    "box",
    "po",
    "pobox",
    "bldg",
    "building",
    "hwy",
    "highway",
    "expressway",
    "route",
    "rte",
    "pkwy",
    "parkway",
    "north",
    "south",
    "east",
    "west",
    "n",
    "s",
    "e",
    "w",
    "no",
    "nr",
    "near",
    "opp",
    "opposite",
    "behind",
    "beside",
}

# Common name tokens that are often non-discriminative across businesses
DEFAULT_NAME_STOPWORDS: Set[str] = {
    "the",
    "and",
    "for",
    "of",
    "in",
    "to",
    "at",
    "a",
    "an",
    "by",
    "with",
    "from",
    "on",
}


@dataclass
class BlockingConfig:
    """Consolidated configuration for candidate blocking strategies and indexing."""

    strategies: List[str] = field(
        default_factory=lambda: [
            "exact_name",
            "name_token",
            "name_prefix",
            "address_conservative",
            "country_scoped_name",
        ]
    )
    min_token_length: int = 3
    prefix_length: int = 4
    max_posting_list_size: int = 500
    min_token_frequency: int = 1
    max_token_frequency: Optional[int] = None
    address_min_token_length: int = 4
    address_stopwords: Set[str] = field(default_factory=lambda: set(DEFAULT_ADDRESS_STOPWORDS))
    name_stopwords: Set[str] = field(default_factory=lambda: set(DEFAULT_NAME_STOPWORDS))
    country_agreement_mode: str = "allow_missing"  # "allow_missing", "strict", "none"
    chunksize: int = 50000
    sample_size: Optional[int] = None
    seed: int = 42
    version: str = BLOCKING_VERSION
    raw_input_prefix: str = "raw/dataset"
    norm_input_prefix: str = "artifacts/normalized/v001"
    output_prefix: str = "artifacts/candidates/block_v001"
    aws_region: str = "ap-southeast-2"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        d = asdict(self)
        d["address_stopwords"] = sorted(list(self.address_stopwords))
        d["name_stopwords"] = sorted(list(self.name_stopwords))
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlockingConfig":
        """Instantiate BlockingConfig from dictionary."""
        d = dict(data)
        if "address_stopwords" in d and isinstance(d["address_stopwords"], (list, set)):
            d["address_stopwords"] = set(d["address_stopwords"])
        if "name_stopwords" in d and isinstance(d["name_stopwords"], (list, set)):
            d["name_stopwords"] = set(d["name_stopwords"])
        return cls(**d)
