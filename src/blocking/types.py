"""Data structures and types for Phase 4 Blocking and Candidate Generation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class CandidatePair:
    """Represents a generated candidate pair between Source 1 and a target entity.

    Attributes:
        source1_entity_id: Identifier of the query entity from Source 1.
        target_entity_id: Identifier of the target candidate from Source 2 or 3.
        target_source: Originating target source ("Source 2" or "Source 3").
        strategies: Tuple of strategy names that generated this candidate pair.
    """

    source1_entity_id: str
    target_entity_id: str
    target_source: str
    strategies: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CandidatePair to dictionary."""
        return {
            "source1_entity_id": self.source1_entity_id,
            "target_entity_id": self.target_entity_id,
            "target_source": self.target_source,
            "strategies": ",".join(self.strategies) if self.strategies else "",
        }


@dataclass
class StrategyMetrics:
    """Metrics for a single blocking strategy or composite strategy."""

    strategy_name: str
    recall: float
    total_candidate_pairs: int
    avg_candidates_per_s1: float
    median_candidates_per_s1: float
    p95_candidates_per_s1: float
    max_candidates_per_s1: float
    candidate_reduction_ratio: float
    total_true_positives: int
    recovered_true_positives: int
    missed_true_positives: int
    s2_recall: float
    s3_recall: float
    runtime_seconds: float
    peak_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert StrategyMetrics to dictionary."""
        return asdict(self)


@dataclass
class ValidationReport:
    """Outcome of Phase 4 blocking validation."""

    status: str  # "PASS" or "FAIL"
    checks: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ValidationReport to dictionary."""
        return asdict(self)
