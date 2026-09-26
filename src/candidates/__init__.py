"""Phase 5 Final Candidate Pipeline package."""

from src.candidates.config import (
    BASELINE_BLOCKING_RECALL,
    BASELINE_CANDIDATE_PAIRS,
    BASELINE_S1_RECORDS,
    BASELINE_S2_RECALL,
    BASELINE_S3_RECALL,
    BASELINE_TARGET_POOL,
    FINAL_CANDIDATE_VERSION,
    CandidatePipelineConfig,
)
from src.candidates.pipeline import CandidatePipeline
from src.candidates.validator import CandidateValidationReport, CandidateValidator

__all__ = [
    "FINAL_CANDIDATE_VERSION",
    "BASELINE_CANDIDATE_PAIRS",
    "BASELINE_S1_RECORDS",
    "BASELINE_TARGET_POOL",
    "BASELINE_BLOCKING_RECALL",
    "BASELINE_S2_RECALL",
    "BASELINE_S3_RECALL",
    "CandidatePipelineConfig",
    "CandidatePipeline",
    "CandidateValidator",
    "CandidateValidationReport",
]
