"""Negative pair sampling strategy for Amazon ML Challenge 2026 Phase 3.

Generates reproducible, informative negative candidate pairs:
1. Hard negatives: Sharing country, lexical tokens, or domain membership, but strictly non-matching.
2. Moderate fallback negatives: Sampled from the broader target entity pool.
3. Guaranteed collision-free: Cross-verified against ground truth positive pairs.
4. Deterministic reproducibility: Guaranteed identical sampling across runs with fixed seed.
"""

from collections import defaultdict
import logging
import random
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from src.utils.logger import setup_logger


class NegativeSampler:
    """Samples reproducible hard and moderate negative pairs with zero ground-truth leakage."""

    def __init__(
        self,
        seed: int = 42,
        hard_negative_ratio: float = 0.5,
        logger: Optional[logging.Logger] = None,
    ):
        """Initializes the negative sampler.

        Args:
            seed: Random seed for deterministic reproducibility.
            hard_negative_ratio: Proportion of negatives to draw from hard candidates (default 0.5).
            logger: Logger instance.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.hard_negative_ratio = hard_negative_ratio
        self.logger = logger or setup_logger(name="negative_sampler")

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Resets RNG to ensure exact deterministic reproducibility."""
        s = seed if seed is not None else self.seed
        self.rng = random.Random(s)

    def sample_negatives_for_source1(
        self,
        s1_id: str,
        num_negatives: int,
        true_matches: Set[str],
        hard_candidates: List[str],
        fallback_candidates: List[str],
    ) -> List[str]:
        """Samples non-matching target IDs for a given Source 1 entity.

        Args:
            s1_id: Source 1 entity ID.
            num_negatives: Target count of negative samples.
            true_matches: Set of authoritative target IDs from ground truth for this s1_id.
            hard_candidates: Target IDs sharing attributes (same country/token) with s1.
            fallback_candidates: Target IDs from the wider source pool.

        Returns:
            List of sampled negative target entity IDs.
        """
        pairs_with_meta = self.sample_negatives_with_metadata(
            s1_id=s1_id,
            num_negatives=num_negatives,
            true_matches=true_matches,
            hard_candidates=hard_candidates,
            fallback_candidates=fallback_candidates,
        )
        return [target_id for target_id, _ in pairs_with_meta]

    def sample_negatives_with_metadata(
        self,
        s1_id: str,
        num_negatives: int,
        true_matches: Set[str],
        hard_candidates: List[str],
        fallback_candidates: List[str],
    ) -> List[Tuple[str, str]]:
        """Samples non-matching target IDs and records the negative strategy used.

        Returns:
            List of (target_id, pair_type) tuples, where pair_type is 'hard_negative' or 'random_negative'.
        """
        sampled: List[Tuple[str, str]] = []
        selected_set: Set[str] = set()

        # 1. Filter out true positive matches from hard candidates
        valid_hard = [c for c in hard_candidates if c not in true_matches]
        num_hard_target = int(num_negatives * self.hard_negative_ratio)

        if valid_hard:
            self.rng.shuffle(valid_hard)
            for c in valid_hard:
                if len(sampled) >= num_hard_target:
                    break
                if c not in selected_set:
                    sampled.append((c, "hard_negative"))
                    selected_set.add(c)

        # 2. Fill remainder with fallback candidates
        needed = num_negatives - len(sampled)
        if needed > 0 and fallback_candidates:
            attempts = 0
            max_attempts = needed * 20
            pool_len = len(fallback_candidates)

            while len(sampled) < num_negatives and attempts < max_attempts:
                attempts += 1
                idx = self.rng.randint(0, pool_len - 1)
                cand = fallback_candidates[idx]
                if cand not in true_matches and cand not in selected_set:
                    sampled.append((cand, "random_negative"))
                    selected_set.add(cand)

        return sampled

    def sample_negatives_by_country(
        self,
        s1_id: str,
        country: Optional[str],
        num_negatives: int,
        true_matches: Set[str],
        country_candidate_pool: Dict[str, List[str]],
        global_fallback_pool: List[str],
        lexical_candidates: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """Samples hard negatives by leveraging same-country and lexical candidate pools.

        Informative/Hard negatives:
        - Targets from the same country (so the model learns entity distinctions within-market).
        - Targets sharing name tokens (if provided).
        Fallback negatives:
        - Uniformly drawn from the global pool when country pool is exhausted.
        """
        hard_candidates: List[str] = []
        if lexical_candidates:
            hard_candidates.extend(lexical_candidates)

        if country and country in country_candidate_pool:
            same_country_targets = country_candidate_pool[country]
            if len(same_country_targets) > 50:
                sample_k = min(50, len(same_country_targets))
                sampled_sub = self.rng.sample(same_country_targets, sample_k)
                hard_candidates.extend(sampled_sub)
            else:
                hard_candidates.extend(same_country_targets)

        return self.sample_negatives_with_metadata(
            s1_id=s1_id,
            num_negatives=num_negatives,
            true_matches=true_matches,
            hard_candidates=hard_candidates,
            fallback_candidates=global_fallback_pool,
        )
