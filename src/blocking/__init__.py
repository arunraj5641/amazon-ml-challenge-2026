"""Phase 4 Blocking and Candidate Generation package."""

from src.blocking.config import (
    BLOCKING_VERSION,
    BlockingConfig,
    DEFAULT_ADDRESS_STOPWORDS,
    DEFAULT_NAME_STOPWORDS,
)
from src.blocking.evaluator import BlockingEvaluator
from src.blocking.generator import CandidateGenerator
from src.blocking.indexes import BlockingIndex
from src.blocking.io import (
    save_all_blocking_artifacts,
    save_candidate_pairs,
    save_json_artifact,
)
from src.blocking.metrics import compute_blocking_metrics
from src.blocking.strategies import (
    AddressConservativeStrategy,
    BlockingStrategy,
    CompositeStrategy,
    CountryScopedNameStrategy,
    ExactNameStrategy,
    NamePrefixStrategy,
    NameTokenStrategy,
    create_default_strategies,
    create_strategy,
)
from src.blocking.recall_evaluator import BlockingRecallEvaluator
from src.blocking.types import CandidatePair, StrategyMetrics, ValidationReport
from src.blocking.validator import BlockingValidator

__all__ = [
    "BLOCKING_VERSION",
    "BlockingConfig",
    "DEFAULT_ADDRESS_STOPWORDS",
    "DEFAULT_NAME_STOPWORDS",
    "CandidatePair",
    "StrategyMetrics",
    "ValidationReport",
    "BlockingIndex",
    "BlockingStrategy",
    "ExactNameStrategy",
    "NameTokenStrategy",
    "NamePrefixStrategy",
    "AddressConservativeStrategy",
    "CountryScopedNameStrategy",
    "CompositeStrategy",
    "create_strategy",
    "create_default_strategies",
    "CandidateGenerator",
    "BlockingEvaluator",
    "BlockingRecallEvaluator",
    "compute_blocking_metrics",
    "BlockingValidator",
    "save_candidate_pairs",
    "save_json_artifact",
    "save_all_blocking_artifacts",
]
