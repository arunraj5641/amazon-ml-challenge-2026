"""Configuration parameters for Phase 5 Candidate Pipeline."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Set

FINAL_CANDIDATE_VERSION = "final_v001"
INPUT_BLOCKING_VERSION = "block_v001"

# Official Phase 4 baseline counts
BASELINE_S1_RECORDS = 2_206_821
BASELINE_TARGET_POOL = 10_320_219
BASELINE_CANDIDATE_PAIRS = 499_912_583
BASELINE_UNIQUE_CANDIDATES = 499_912_583
BASELINE_BLOCKING_RECALL = 0.881188
BASELINE_S2_RECALL = 0.885255
BASELINE_S3_RECALL = 0.877380


@dataclass
class CandidatePipelineConfig:
    """Configuration for Phase 5 candidate finalization and validation."""

    version: str = FINAL_CANDIDATE_VERSION
    input_blocking_version: str = INPUT_BLOCKING_VERSION
    expected_candidate_count: int = BASELINE_CANDIDATE_PAIRS
    expected_s1_records: int = BASELINE_S1_RECORDS
    expected_target_pool: int = BASELINE_TARGET_POOL
    chunksize: int = 50_000
    allow_deduplication: bool = True
    strict_ordering: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to a serializable dictionary."""
        return asdict(self)
