"""Feature extraction logic for pairwise entity comparison."""

from dataclasses import dataclass
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from src.features.config import FEATURE_NAMES

# Regex for extracting numeric digits / tokens from address
NUMBER_PATTERN = re.compile(r"\b\d+\b")


def extract_3grams(text: str) -> FrozenSet[str]:
    """Extracts character 3-grams from normalized text."""
    if not text or len(text) < 3:
        return frozenset()
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))


@dataclass(slots=True, frozen=True)
class EntityPreprocessedRecord:
    """Preprocessed compact entity record for rapid pairwise feature computation."""

    entity_id: str
    name_norm: str
    name_core: str
    name_alnum: str
    addr_norm: str
    addr_alnum: str
    country_norm: str
    name_tokens: FrozenSet[str]
    name_qgrams: FrozenSet[str]
    addr_tokens: FrozenSet[str]
    addr_numbers: FrozenSet[str]
    raw_name_missing: bool
    raw_addr_missing: bool
    raw_country_missing: bool

    @classmethod
    def from_row_dict(cls, row: Dict[str, Any]) -> "EntityPreprocessedRecord":
        """Constructs an EntityPreprocessedRecord from a normalized TSV row dictionary."""
        e_id = str(row.get("entity_id", "")).strip()

        # Raw fields
        raw_name = str(row.get("business_name", "")).strip()
        raw_addr = str(row.get("business_address", "")).strip()
        raw_country = str(row.get("country", "")).strip()

        raw_name_missing = not raw_name or raw_name.lower() in ("nan", "null", "none", "<na>")
        raw_addr_missing = not raw_addr or raw_addr.lower() in ("nan", "null", "none", "<na>")
        raw_country_missing = not raw_country or raw_country.lower() in ("nan", "null", "none", "<na>")

        # Normalized fields
        name_norm = str(row.get("business_name_normalized", "")).strip() if not pd.isna(row.get("business_name_normalized")) else ""
        name_core = str(row.get("business_name_core", "")).strip() if not pd.isna(row.get("business_name_core")) else ""
        name_alnum = str(row.get("business_name_alnum", "")).strip() if not pd.isna(row.get("business_name_alnum")) else ""
        addr_norm = str(row.get("business_address_normalized", "")).strip() if not pd.isna(row.get("business_address_normalized")) else ""
        addr_alnum = str(row.get("business_address_alnum", "")).strip() if not pd.isna(row.get("business_address_alnum")) else ""
        country_norm = str(row.get("country_normalized", "")).strip() if not pd.isna(row.get("country_normalized")) else ""

        # Precompute tokens and qgrams
        name_tokens = frozenset(name_norm.split()) if name_norm else frozenset()
        name_qgrams = extract_3grams(name_norm)
        addr_tokens = frozenset(addr_norm.split()) if addr_norm else frozenset()
        addr_numbers = frozenset(NUMBER_PATTERN.findall(addr_norm)) if addr_norm else frozenset()

        return cls(
            entity_id=e_id,
            name_norm=name_norm,
            name_core=name_core,
            name_alnum=name_alnum,
            addr_norm=addr_norm,
            addr_alnum=addr_alnum,
            country_norm=country_norm,
            name_tokens=name_tokens,
            name_qgrams=name_qgrams,
            addr_tokens=addr_tokens,
            addr_numbers=addr_numbers,
            raw_name_missing=raw_name_missing,
            raw_addr_missing=raw_addr_missing,
            raw_country_missing=raw_country_missing,
        )


class PairFeatureExtractor:
    """Computes deterministic numerical features between two preprocessed entity records."""

    @staticmethod
    def extract_features(
        s1: EntityPreprocessedRecord,
        target: EntityPreprocessedRecord,
        target_source: str = "",
        strategies_str: str = "",
    ) -> List[float]:
        """Extracts all 51 features matching FEATURE_NAMES order."""
        # 1. Name features
        n1 = s1.name_norm
        n2 = target.name_norm
        exact_name = 1.0 if (n1 and n1 == n2) else 0.0
        exact_core = 1.0 if (s1.name_core and s1.name_core == target.name_core) else 0.0
        exact_alnum = 1.0 if (s1.name_alnum and s1.name_alnum == target.name_alnum) else 0.0

        len_n1 = len(n1)
        len_n2 = len(n2)
        name_char_len_diff = float(abs(len_n1 - len_n2))
        max_name_len = max(len_n1, len_n2)
        name_char_len_ratio = (min(len_n1, len_n2) / max_name_len) if max_name_len > 0 else 0.0

        nt1 = s1.name_tokens
        nt2 = target.name_tokens
        name_u = nt1 | nt2
        name_i = nt1 & nt2
        name_token_jaccard = (len(name_i) / len(name_u)) if name_u else 0.0
        name_token_overlap_count = float(len(name_i))
        name_token_dice = (2.0 * len(name_i) / (len(nt1) + len(nt2))) if (len(nt1) + len(nt2)) > 0 else 0.0
        min_nt = min(len(nt1), len(nt2))
        name_token_containment = (len(name_i) / min_nt) if min_nt > 0 else 0.0

        name_prefix_3 = 1.0 if (len_n1 >= 3 and len_n2 >= 3 and n1[:3] == n2[:3]) else 0.0
        name_prefix_5 = 1.0 if (len_n1 >= 5 and len_n2 >= 5 and n1[:5] == n2[:5]) else 0.0

        nq1 = s1.name_qgrams
        nq2 = target.name_qgrams
        nq_u = nq1 | nq2
        nq_i = nq1 & nq2
        name_char_qgram_jaccard = (len(nq_i) / len(nq_u)) if nq_u else 0.0

        # 2. Address features
        ad1 = s1.addr_norm
        ad2 = target.addr_norm
        exact_addr = 1.0 if (ad1 and ad1 == ad2) else 0.0
        exact_addr_alnum = 1.0 if (s1.addr_alnum and s1.addr_alnum == target.addr_alnum) else 0.0

        len_ad1 = len(ad1)
        len_ad2 = len(ad2)
        addr_char_len_diff = float(abs(len_ad1 - len_ad2))
        max_addr_len = max(len_ad1, len_ad2)
        addr_char_len_ratio = (min(len_ad1, len_ad2) / max_addr_len) if max_addr_len > 0 else 0.0

        at1 = s1.addr_tokens
        at2 = target.addr_tokens
        addr_u = at1 | at2
        addr_i = at1 & at2
        addr_token_jaccard = (len(addr_i) / len(addr_u)) if addr_u else 0.0
        addr_token_overlap_count = float(len(addr_i))
        addr_token_dice = (2.0 * len(addr_i) / (len(at1) + len(at2))) if (len(at1) + len(at2)) > 0 else 0.0
        min_at = min(len(at1), len(at2))
        addr_token_containment = (len(addr_i) / min_at) if min_at > 0 else 0.0

        num1 = s1.addr_numbers
        num2 = target.addr_numbers
        num_u = num1 | num2
        num_i = num1 & num2
        addr_num_overlap_count = float(len(num_i))
        addr_num_jaccard = (len(num_i) / len(num_u)) if num_u else 0.0

        # 3. Country features
        c1 = s1.country_norm
        c2 = target.country_norm
        country_exact = 1.0 if (c1 and c2 and c1 == c2) else 0.0
        country_mismatch = 1.0 if (c1 and c2 and c1 != c2) else 0.0
        country_missing_either = 1.0 if (not c1 or not c2) else 0.0
        country_missing_both = 1.0 if (not c1 and not c2) else 0.0

        # 4. Source & Strategy features
        is_source2 = 1.0 if (target_source == "Source 2" or target.entity_id.lower().startswith(("s2_", "s2-"))) else 0.0
        is_source3 = 1.0 if (target_source == "Source 3" or target.entity_id.lower().startswith(("s3_", "s3-"))) else 0.0

        strat_set = set(s.strip() for s in strategies_str.split(",") if s.strip()) if strategies_str else set()
        strategy_count = float(len(strat_set)) if strat_set else 1.0
        strat_exact_name = 1.0 if "exact_name" in strat_set else 0.0
        strat_name_token = 1.0 if "name_token" in strat_set else 0.0
        strat_name_prefix = 1.0 if "name_prefix" in strat_set else 0.0
        strat_addr_conservative = 1.0 if "address_conservative" in strat_set else 0.0
        strat_country_scoped = 1.0 if "country_scoped_name" in strat_set else 0.0
        strat_composite = 1.0 if "composite" in strat_set else 0.0

        # 5. Missingness & Quality features
        s1_name_missing = 1.0 if s1.raw_name_missing else 0.0
        target_name_missing = 1.0 if target.raw_name_missing else 0.0
        s1_address_missing = 1.0 if s1.raw_addr_missing else 0.0
        target_address_missing = 1.0 if target.raw_addr_missing else 0.0
        s1_country_missing = 1.0 if s1.raw_country_missing else 0.0
        target_country_missing = 1.0 if target.raw_country_missing else 0.0
        both_address_missing = 1.0 if (s1.raw_addr_missing and target.raw_addr_missing) else 0.0

        avail_s1 = int(not s1.raw_name_missing) + int(not s1.raw_addr_missing) + int(not s1.raw_country_missing)
        avail_target = int(not target.raw_name_missing) + int(not target.raw_addr_missing) + int(not target.raw_country_missing)
        total_available_fields = float(avail_s1 + avail_target)

        # 6. Entity length & token count features
        s1_name_char_len = float(len_n1)
        target_name_char_len = float(len_n2)
        s1_name_token_count = float(len(nt1))
        target_name_token_count = float(len(nt2))
        s1_address_char_len = float(len_ad1)
        target_address_char_len = float(len_ad2)
        s1_address_token_count = float(len(at1))
        target_address_token_count = float(len(at2))

        return [
            # 1. Name
            exact_name,
            exact_core,
            exact_alnum,
            name_char_len_diff,
            name_char_len_ratio,
            name_token_jaccard,
            name_token_overlap_count,
            name_token_dice,
            name_token_containment,
            name_prefix_3,
            name_prefix_5,
            name_char_qgram_jaccard,
            # 2. Address
            exact_addr,
            exact_addr_alnum,
            addr_char_len_diff,
            addr_char_len_ratio,
            addr_token_jaccard,
            addr_token_overlap_count,
            addr_token_dice,
            addr_token_containment,
            addr_num_overlap_count,
            addr_num_jaccard,
            # 3. Country
            country_exact,
            country_mismatch,
            country_missing_either,
            country_missing_both,
            # 4. Source & Strategy
            is_source2,
            is_source3,
            strategy_count,
            strat_exact_name,
            strat_name_token,
            strat_name_prefix,
            strat_addr_conservative,
            strat_country_scoped,
            strat_composite,
            # 5. Missingness & Quality
            s1_name_missing,
            target_name_missing,
            s1_address_missing,
            target_address_missing,
            s1_country_missing,
            target_country_missing,
            both_address_missing,
            total_available_fields,
            # 6. Length & Token counts
            s1_name_char_len,
            target_name_char_len,
            s1_name_token_count,
            target_name_token_count,
            s1_address_char_len,
            target_address_char_len,
            s1_address_token_count,
            target_address_token_count,
        ]
