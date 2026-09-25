"""Deterministic, Unicode-aware text normalization for business entity resolution.

Provides normalization functions for:
- business_name (normalized, core without legal suffixes, alphanumeric)
- business_address (normalized, alphanumeric)
- country (open-set canonicalization with conservative aliases)

All functions are:
- Safe for None/empty/whitespace inputs
- Deterministic and idempotent: f(f(x)) == f(x)
- Free from external API or ML model dependencies
- Unicode NFKC normalized
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

# Unicode normalization form
UNICODE_NORMALIZATION_FORM = "NFKC"

# Recognized legal suffixes for business entity resolution (case-insensitive)
# Ordered from longest/multi-word to single-word to prevent partial prefix matching
RECOGNIZED_LEGAL_SUFFIXES: List[str] = [
    # Multi-word suffixes
    "public limited company",
    "limited liability company",
    "limited liability partnership",
    "joint stock company",
    "private limited",
    "pvt limited",
    "pvt ltd",
    "pte ltd",
    "pty ltd",
    "co ltd",
    "sociedad anonima",
    "societe anonyme",
    "societa a responsabilita limitata",
    "sociedad de responsabilidad limitada",
    # Single-word suffixes & abbreviations
    "incorporated",
    "corporation",
    "limited",
    "company",
    "corp",
    "inc",
    "ltd",
    "llc",
    "llp",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "sarl",
    "bv",
    "nv",
    "spa",
    "srl",
    "ab",
    "oy",
    "as",
    "co",
    "cie",
]

# Precompiled regex pattern matching recognized legal suffixes at the end of a string
# Uses word boundaries to ensure we only match standalone suffix words at the end
_LEGAL_SUFFIX_PATTERN = re.compile(
    r"(?:^|\s+)(" + "|".join(re.escape(s) for s in RECOGNIZED_LEGAL_SUFFIXES) + r")(?:\s+|$)",
    flags=re.IGNORECASE,
)

# Open-set country aliases mapping to standardized lowercase 2-letter ISO codes.
# Note: Unrecognized countries are NOT rejected; they pass through as cleaned lowercase text.
COUNTRY_ALIASES: Dict[str, str] = {
    # United States
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
    "u.s.": "us",
    "u.s.a.": "us",
    "us": "us",
    # India
    "india": "in",
    "ind": "in",
    "in": "in",
    # France
    "france": "fr",
    "fra": "fr",
    "fr": "fr",
    # Germany
    "germany": "de",
    "deutschland": "de",
    "deu": "de",
    "ger": "de",
    "de": "de",
    # United Kingdom
    "united kingdom": "gb",
    "great britain": "gb",
    "uk": "gb",
    "gbr": "gb",
    "gb": "gb",
    # Canada
    "canada": "ca",
    "can": "ca",
    "ca": "ca",
    # Australia
    "australia": "au",
    "aus": "au",
    "au": "au",
    # Japan
    "japan": "jp",
    "jpn": "jp",
    "jp": "jp",
    # Sweden
    "sweden": "se",
    "sverige": "se",
    "swe": "se",
    "se": "se",
    # Singapore
    "singapore": "sg",
    "sgp": "sg",
    "sg": "sg",
    # Brazil
    "brazil": "br",
    "brasil": "br",
    "bra": "br",
    "br": "br",
    # Italy
    "italy": "it",
    "italia": "it",
    "ita": "it",
    "it": "it",
    # Spain
    "spain": "es",
    "espana": "es",
    "esp": "es",
    "es": "es",
    # Netherlands
    "netherlands": "nl",
    "holland": "nl",
    "nld": "nl",
    "nl": "nl",
    # China
    "china": "cn",
    "prc": "cn",
    "chn": "cn",
    "cn": "cn",
    # Mexico
    "mexico": "mx",
    "mex": "mx",
    "mx": "mx",
}


def _clean_unicode_and_whitespace(text: Any) -> str:
    """Safely converts input to NFKC normalized, lowercase, whitespace-collapsed string."""
    if text is None:
        return ""
    s = str(text)
    if not s or s.lower() in ("nan", "null", "none", "<na>"):
        return ""

    # NFKC normalizes characters, ligatures, fullwidth/halfwidth
    norm = unicodedata.normalize(UNICODE_NORMALIZATION_FORM, s)
    # Case fold to lowercase
    lower = norm.lower()
    # Replace non-breaking spaces, control characters, tabs, newlines
    cleaned = re.sub(r"[\t\r\n\u00a0\u200b\ufeff]", " ", lower)
    # Collapse multiple whitespace
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def normalize_business_name(text: Optional[str]) -> str:
    """Conservative business name normalization.

    Operations:
    1. Unicode NFKC normalization and lowercase conversion.
    2. Normalize ampersands (& -> and) and plus signs (+ -> and).
    3. Remove abbreviation dots (e.g. 'pvt. ltd.' -> 'pvt ltd', 'a.b.c.' -> 'abc').
    4. Normalize punctuation separators (commas, semicolons, brackets, quotes) to spaces.
    5. Preserve hyphens between alphanumeric characters (e.g. 'wal-mart', 'hewlett-packard').
    6. Collapse whitespace.

    Args:
        text: Raw business name.

    Returns:
        Conservatively normalized business name string.
    """
    s = _clean_unicode_and_whitespace(text)
    if not s:
        return ""

    # Standardize ampersands and plus signs
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"\+", " and ", s)

    # Normalize acronym dots: single letter dots 'a.b.c.' -> 'abc'
    s = re.sub(r"(?<=\b\w)\.(?=\w\b)", "", s)
    # Remove dots after words or at end: 'pvt. ltd.' -> 'pvt ltd'
    s = re.sub(r"(?<=\w)\.(?=\s|$)", "", s)
    # Remove standalone dots
    s = re.sub(r"\s+\.\s+", " ", s)

    # Replace punctuation that acts as separators with space, but preserve hyphens and apostrophes
    s = re.sub(r"[,;:\/\\\|\(\)\[\]\{\}\"~`!@#\$%\^*_=\?<>]+", " ", s)

    # Clean isolated hyphens or apostrophes not joining words
    s = re.sub(r"(?<!\w)-|-(?!\w)", " ", s)
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)

    # Collapse whitespace and strip
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_business_name_core(name_normalized: str) -> str:
    """Extracts core business name by stripping recognized legal suffixes from the end.

    Only removes suffixes that occur at the end of the normalized name.
    If stripping all suffixes would result in an empty string, the original normalized
    name is preserved.

    Args:
        name_normalized: Pre-normalized business name string.

    Returns:
        Core business name string.
    """
    if not name_normalized:
        return ""

    curr = name_normalized
    # Iteratively strip legal suffixes from the tail of the name
    while True:
        stripped = False
        for suffix in RECOGNIZED_LEGAL_SUFFIXES:
            # Check if string ends with this suffix as a distinct token (preceded by whitespace or hyphen)
            pattern = r"(?:^|[\s\-])" + re.escape(suffix) + r"$"
            match = re.search(pattern, curr, flags=re.IGNORECASE)
            if match:
                trimmed = curr[: match.start()].strip(" -")
                # Do not reduce to empty string
                if trimmed:
                    curr = trimmed
                    stripped = True
                    break
        if not stripped:
            break

    return curr if curr else name_normalized


def normalize_business_name_alnum(text: Optional[str]) -> str:
    """Produces alphanumeric representation of business name.

    Retains all Unicode letters and numbers across global scripts (Latin,
    Cyrillic, CJK, Arabic, Devanagari, etc.) and combining marks, replacing
    punctuation and symbols with spaces.

    Args:
        text: Raw or normalized business name.

    Returns:
        Unicode-aware alphanumeric business name string.
    """
    s = normalize_business_name(text)
    if not s:
        return ""
    # Retain all Unicode letters, numbers, and combining marks
    chars = "".join(
        c if (c.isalnum() or unicodedata.category(c).startswith("M")) else " " for c in s
    )
    return re.sub(r"\s+", " ", chars).strip()


def normalize_business_address(text: Optional[str]) -> str:
    """Conservative business address normalization.

    Operations:
    1. Unicode NFKC normalization and lowercase conversion.
    2. Replace address separators (commas, semicolons, hashes, colons, slashes) with space.
    3. Remove abbreviation dots (e.g. 'st.' -> 'st', 'no.' -> 'no', 'rd.' -> 'rd').
    4. Preserve hyphens between unit/building digits and letters (e.g. '10-a', 'suite 100-b').
    5. Collapse whitespace.

    Args:
        text: Raw business address.

    Returns:
        Conservatively normalized business address string.
    """
    s = _clean_unicode_and_whitespace(text)
    if not s:
        return ""

    # Remove abbreviation dots (e.g. 'st.', 'rd.', 'ave.', 'no.')
    s = re.sub(r"(?<=\w)\.(?=\s|$)", "", s)
    # Remove standalone dots
    s = re.sub(r"\s+\.\s+", " ", s)

    # Standardize separators to spaces (commas, colons, semicolons, hashes, slashes, brackets)
    s = re.sub(r"[,;:\/\\\|\(\)\[\]\{\}\"~`!@#\$%\^&*_=\?<>]+", " ", s)

    # Clean isolated hyphens or apostrophes not joining words/digits across all scripts
    s = re.sub(r"(?<!\w)-|-(?!\w)", " ", s)
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)

    # Collapse whitespace and strip
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_business_address_alnum(text: Optional[str]) -> str:
    """Produces alphanumeric representation of business address.

    Retains all Unicode letters and numbers across global scripts and
    combining marks, replacing punctuation and symbols with spaces.

    Args:
        text: Raw or normalized business address.

    Returns:
        Unicode-aware alphanumeric business address string.
    """
    s = normalize_business_address(text)
    if not s:
        return ""
    chars = "".join(
        c if (c.isalnum() or unicodedata.category(c).startswith("M")) else " " for c in s
    )
    return re.sub(r"\s+", " ", chars).strip()


def normalize_country(text: Optional[str]) -> str:
    """Open-set country normalization.

    Operations:
    1. Unicode NFKC normalization and lowercase conversion.
    2. Strip extraneous whitespace and punctuation.
    3. Map well-known aliases to canonical 2-letter ISO codes where available.
    4. Pass through any other country names as clean lowercase strings.
       (Open-set principle: NEVER rejects or discards unknown countries).

    Args:
        text: Raw country value.

    Returns:
        Normalized country string (or '' if empty/null).
    """
    s = _clean_unicode_and_whitespace(text)
    if not s:
        return ""

    # Strip surrounding punctuation like 'u.s.' -> 'us' or '(france)' -> 'france'
    s = s.strip(" .,;:-_()[]'\"")
    # Collapse internal spaces
    s = re.sub(r"\s+", " ", s)

    # Check alias dictionary
    if s in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[s]

    # Try removing internal dots (e.g. 'u.s.' -> 'us', 'u.k.' -> 'uk')
    s_nodots = s.replace(".", "")
    if s_nodots in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[s_nodots]

    # Open-set: return the cleaned lowercase string directly
    return s
