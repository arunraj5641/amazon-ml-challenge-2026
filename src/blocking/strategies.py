"""Modular blocking strategy interface and strategy family implementations."""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional, Set

from src.blocking.config import BlockingConfig
from src.normalization.normalizer import (
    extract_business_name_core,
    normalize_business_address_alnum,
    normalize_business_name_alnum,
    normalize_country,
)


def get_field_or_derive(record: Dict[str, Any], field_name: str) -> str:
    """Safely retrieves a field or derives its normalized representation if absent."""
    val = record.get(field_name)
    if val is not None and str(val).strip() and str(val).strip().lower() not in ("nan", "null", "none", "<na>"):
        return str(val).strip()

    # Derived fallbacks if working on raw tables
    if field_name == "business_name_core":
        raw_name = record.get("business_name", "")
        if raw_name:
            norm_name = normalize_business_name_alnum(str(raw_name))
            return extract_business_name_core(norm_name)
        return ""

    if field_name == "business_name_alnum":
        raw_name = record.get("business_name", "")
        return normalize_business_name_alnum(str(raw_name)) if raw_name else ""

    if field_name == "business_address_alnum":
        raw_addr = record.get("business_address", "")
        return normalize_business_address_alnum(str(raw_addr)) if raw_addr else ""

    if field_name == "country_normalized":
        raw_country = record.get("country", "")
        return normalize_country(str(raw_country)) if raw_country else ""

    return ""


class BlockingStrategy(ABC):
    """Abstract base class for all blocking strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this strategy."""
        pass

    @abstractmethod
    def generate_keys(self, record: Dict[str, Any]) -> List[str]:
        """Generate blocking keys for a single business entity record.

        Args:
            record: Dict containing normalized (or raw) entity attributes.

        Returns:
            List of blocking key strings.
        """
        pass


class ExactNameStrategy(BlockingStrategy):
    """Exact normalized business name blocking strategy.

    Matches entities having identical core business names after legal suffix
    stripping and alphanumeric canonicalization.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()

    @property
    def name(self) -> str:
        return "exact_name"

    def generate_keys(self, record: Dict[str, Any]) -> List[str]:
        core_name = get_field_or_derive(record, "business_name_core")
        if not core_name:
            core_name = get_field_or_derive(record, "business_name_alnum")

        if not core_name:
            return []

        # Canonical key format
        return [f"exact:{core_name}"]


class NameTokenStrategy(BlockingStrategy):
    """Name token blocking strategy.

    Indexes entities by individual discriminative name tokens to capture
    word-order variation, additions, and partial name overlap.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        self.min_length = self.config.min_token_length
        self.stopwords = self.config.name_stopwords

    @property
    def name(self) -> str:
        return "name_token"

    def generate_keys(self, record: Dict[str, Any]) -> List[str]:
        core_name = get_field_or_derive(record, "business_name_core")
        if not core_name:
            core_name = get_field_or_derive(record, "business_name_alnum")

        if not core_name:
            return []

        tokens = re.findall(r"\w+", core_name)
        keys: List[str] = []
        seen: Set[str] = set()

        for t in tokens:
            if len(t) < self.min_length:
                continue
            if t in self.stopwords:
                continue
            if t not in seen:
                seen.add(t)
                keys.append(f"tok:{t}")

        return keys


class NamePrefixStrategy(BlockingStrategy):
    """Character prefix / abbreviation blocking strategy.

    Indexes entities by character prefixes of significant tokens to handle
    minor typos, stem variations, and truncated business names.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        self.prefix_len = self.config.prefix_length
        self.stopwords = self.config.name_stopwords

    @property
    def name(self) -> str:
        return "name_prefix"

    def generate_keys(self, record: Dict[str, Any]) -> List[str]:
        core_name = get_field_or_derive(record, "business_name_core")
        if not core_name:
            core_name = get_field_or_derive(record, "business_name_alnum")

        if not core_name:
            return []

        tokens = re.findall(r"\w+", core_name)
        keys: List[str] = []
        seen: Set[str] = set()

        for t in tokens:
            if len(t) < self.prefix_len:
                continue
            if t in self.stopwords:
                continue
            prefix = t[: self.prefix_len]
            if prefix not in seen:
                seen.add(prefix)
                keys.append(f"pre{self.prefix_len}:{prefix}")

        return keys


class AddressConservativeStrategy(BlockingStrategy):
    """Conservative address-derived blocking strategy.

    Generates high-precision address keys (street number + street name token,
    postal/PIN codes, and specific non-stopword tokens) while suppressing
    pathological general address tokens (e.g., 'street', 'road', 'suite').
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        self.min_length = self.config.address_min_token_length
        self.stopwords = self.config.address_stopwords

    @property
    def name(self) -> str:
        return "address_conservative"

    def generate_keys(self, record: Dict[str, Any]) -> List[str]:
        addr = get_field_or_derive(record, "business_address_alnum")
        if not addr:
            return []

        tokens = re.findall(r"\w+", addr)
        if not tokens:
            return []

        keys: List[str] = []
        seen: Set[str] = set()

        # 1. Street number + first street token combination (e.g. "100" + "market" -> "addr_num:100_market")
        for i, t in enumerate(tokens):
            if t.isdigit() and len(t) <= 6:
                # Look for adjacent non-digit, non-stopword token
                next_tok = None
                if i + 1 < len(tokens):
                    cand = tokens[i + 1]
                    if not cand.isdigit() and cand not in self.stopwords and len(cand) >= 3:
                        next_tok = cand
                elif i > 0:
                    cand = tokens[i - 1]
                    if not cand.isdigit() and cand not in self.stopwords and len(cand) >= 3:
                        next_tok = cand

                if next_tok:
                    k = f"addr_num:{t}_{next_tok}"
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)

            # 2. Postal / PIN code detection (5 or 6 digit standalone numbers)
            if t.isdigit() and len(t) in (5, 6):
                k = f"addr_zip:{t}"
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

            # 3. Conservative discriminative address token (length >= min_length, not stopword, not pure digits)
            elif not t.isdigit() and len(t) >= self.min_length and t not in self.stopwords:
                k = f"addr_tok:{t}"
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

        return keys


class CountryScopedNameStrategy(BlockingStrategy):
    """Country-aware name blocking strategy.

    Combines open-set country representation with exact core name and primary
    tokens. Supports open-set countries (including France, etc.) and safely
    handles missing/unknown countries without dropping valid matches.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        self.stopwords = self.config.name_stopwords

    @property
    def name(self) -> str:
        return "country_scoped_name"

    def generate_keys(self, record: Dict[str, Any]) -> List[str]:
        core_name = get_field_or_derive(record, "business_name_core")
        if not core_name:
            core_name = get_field_or_derive(record, "business_name_alnum")

        if not core_name:
            return []

        country = get_field_or_derive(record, "country_normalized")
        geo_tag = country if country else "unknown"

        keys: List[str] = [f"c_name:{geo_tag}:{core_name}"]

        # Also country-scoped significant tokens
        tokens = re.findall(r"\w+", core_name)
        seen: Set[str] = set()
        for t in tokens:
            if len(t) < self.config.min_token_length or t in self.stopwords:
                continue
            if t not in seen:
                seen.add(t)
                keys.append(f"c_tok:{geo_tag}:{t}")

        return keys


class CompositeStrategy(BlockingStrategy):
    """Composite blocking strategy uniting multiple modular sub-strategies.

    Gathers keys across all enabled sub-strategies. Enables multi-angle
    candidate generation where different strategies catch different match modalities.
    """

    def __init__(
        self,
        strategies: List[BlockingStrategy],
        strategy_name: str = "composite",
    ):
        self._strategies = strategies
        self._name = strategy_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def sub_strategies(self) -> List[BlockingStrategy]:
        return list(self._strategies)

    def generate_keys(self, record: Dict[str, Any]) -> List[str]:
        all_keys: List[str] = []
        seen: Set[str] = set()
        for strat in self._strategies:
            keys = strat.generate_keys(record)
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        return all_keys

    def generate_keys_with_strategy_map(self, record: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generates keys grouped by sub-strategy name."""
        res: Dict[str, List[str]] = {}
        for strat in self._strategies:
            res[strat.name] = strat.generate_keys(record)
        return res


def create_strategy(name: str, config: Optional[BlockingConfig] = None) -> BlockingStrategy:
    """Factory creating strategy by name."""
    cfg = config or BlockingConfig()
    name_clean = name.strip().lower()

    if name_clean == "exact_name":
        return ExactNameStrategy(cfg)
    elif name_clean == "name_token":
        return NameTokenStrategy(cfg)
    elif name_clean == "name_prefix":
        return NamePrefixStrategy(cfg)
    elif name_clean in ("address_conservative", "address", "address_token"):
        return AddressConservativeStrategy(cfg)
    elif name_clean in ("country_scoped_name", "country_aware", "country_name"):
        return CountryScopedNameStrategy(cfg)
    else:
        raise ValueError(f"Unknown blocking strategy name: '{name}'")


def create_default_strategies(config: Optional[BlockingConfig] = None) -> Dict[str, BlockingStrategy]:
    """Returns a dictionary of all individual strategy instances and a composite strategy."""
    cfg = config or BlockingConfig()
    sub_strats = [
        ExactNameStrategy(cfg),
        NameTokenStrategy(cfg),
        NamePrefixStrategy(cfg),
        AddressConservativeStrategy(cfg),
        CountryScopedNameStrategy(cfg),
    ]
    strats: Dict[str, BlockingStrategy] = {s.name: s for s in sub_strats}
    strats["composite"] = CompositeStrategy(sub_strats)
    return strats
