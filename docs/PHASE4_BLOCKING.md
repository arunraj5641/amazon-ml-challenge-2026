# Phase 4: Candidate Blocking & Filtering Architecture

## 1. Overview & Problem Definition

In the **Amazon ML Challenge 2026 Business Entity Resolution** task, our goal is to link each query entity in **Source 1** to its corresponding matching entities in **Source 2** and **Source 3**. Matches can be:
- **Zero matches** (singletons: S1 entity does not exist in target sources)
- **One match** (1-to-1 link to an entity in S2 or S3)
- **Multiple matches** (1-to-many links to multiple entities across S2 and/or S3)

### The Scale Problem: Why Blocking is Essential
In the competition dataset, target sources contain millions of listings:
- $|S1| \approx 10^6+$
- $|S2| + |S3| \approx 10^7+$

A naive Cartesian product ($S1 \times (S2 \cup S3)$) would require:
$$\approx 10^6 \times 10^7 = 10^{13} \text{ (10 trillion pair comparisons)}$$

Calculating computationally expensive multi-field similarities (Jaro-Winkler, Levenshtein, character n-gram cosine, deep transformer cross-encoders) over 10 trillion pairs is computationally infeasible.

### The Role of Blocking (Candidate Generation)
Blocking reduces the comparison space by orders of magnitude:
```
Source 1 (Query)
       ↓
Blocking Key Generation
       ↓
Inverted Index Lookup (Target Pools S2 & S3)
       ↓
Candidate Set (~10-100 candidates per S1)
       ↓
Phase 6: Expensive Feature Computation
       ↓
Phase 7: Machine Learning Matcher
```

> **Critical ER Principle**:
> Blocking must prioritize **Candidate Recall First**.
> Any true match missed at the blocking stage is a false negative that can **never** be recovered by downstream feature engineering or ML classifiers.

---

## 2. Modular Strategy Architecture

Phase 4 defines a modular, extensible strategy architecture rooted in `BlockingStrategy`:

```
                 BlockingStrategy (ABC)
                           │
      ┌────────────┬───────┴────────┬─────────────┬──────────────────────┐
      ▼            ▼                ▼             ▼                      ▼
  ExactName    NameToken        NamePrefix    AddressConservative    CountryScopedName
      │            │                │             │                      │
      └────────────┴───────┬────────┴─────────────┴──────────────────────┘
                           ▼
                    CompositeStrategy (Union)
```

Each strategy implements `generate_keys(record: Dict[str, Any]) -> List[str]`. This modularity allows:
1. Benchmarking every strategy family in isolation against ground truth.
2. Observing complementary match coverage across modalities.
3. Combining strategies cleanly into a `CompositeStrategy` without pipeline rewrites.

---

## 3. Strategy Families

### A. Exact Normalized Name (`exact_name`)
- **Key Format**: `exact:<business_name_core>`
- **Design**: Uses normalized core business names with recognized legal suffixes (e.g., Corp, Inc, Ltd, Pvt Ltd, GmbH) cleanly removed from the end.
- **Strengths**: Ultra-high precision, zero false-positive explosion.
- **Limitations**: Misses word-order variations and minor typos.

### B. Name Token Blocking (`name_token`)
- **Key Format**: `tok:<token>`
- **Design**: Tokenizes core names, filtering tokens by minimum length (`min_token_length >= 3`) and removing grammatical stopwords (`the`, `and`, `for`, `of`, `in`, `to`, `at`, etc.).
- **Strengths**: Captures word-order variations (e.g., *"Apex Electronics"* vs. *"Electronics Apex"*).
- **Protections**: Guarded by `max_posting_list_size` to suppress pathological common tokens.

### C. Name Prefix / Abbreviation Blocking (`name_prefix`)
- **Key Format**: `pre4:<prefix>` (default length 4)
- **Design**: Extracts fixed-length prefixes from significant business name tokens.
- **Strengths**: Captures typos, phonetic variations, truncations, and stem variations (e.g., *"Internat"* vs. *"International"*, *"Enginering"* vs. *"Engineering"*).

### D. Conservative Address-Derived Blocking (`address_conservative`)
- **Key Format**:
  - Street Number + Name: `addr_num:<number>_<token>` (e.g., `addr_num:100_market`)
  - Standalone Postal/ZIP Code: `addr_zip:<postal_code>` (e.g., `addr_zip:94105`, `addr_zip:560001`)
  - Discriminative Non-Stopword Address Tokens: `addr_tok:<token>` (length $\ge 4$)
- **Pathological Token Suppression**: Standalone general address tokens (`st`, `street`, `rd`, `road`, `ave`, `avenue`, `suite`, `floor`, `blvd`, `box`, `po`, etc.) are strictly stripped from standalone indexing.

### E. Country-Scoped Blocking (`country_scoped_name`)
- **Key Format**: `c_name:<country>:<core_name>`, `c_tok:<country>:<token>`
- **Open-Set Support**: Supports any country representation (canonicalized via `normalize_country`), including France (`fr`), Germany (`de`), UK (`gb`), Sweden (`se`), etc. Never assumes only US/India.
- **Unknown Country Safe**: Entities lacking country information are mapped to `unknown` to ensure they are never dropped or penalized.

### F. Composite Blocking (`composite`)
- **Design**: Evaluates the union of keys generated by active strategies.
- **Traceability**: Candidate pairs retain the complete list of matching strategies that triggered the retrieval (`CandidatePair.strategies`).

---

## 4. Inverted Index Design & Memory Consciousness

Target pools contain millions of entities. `BlockingIndex` is engineered for memory safety:
1. **Zero Raw Record Buffering**: Does not store full DataFrames or raw strings in index memory. Only stores posting lists: `key -> list of target_id`.
2. **Frequency Guards (`max_posting_list_size`)**:
   - During `finalize()`, any blocking key with posting list length $> \text{max\_posting\_list\_size}$ (e.g., 500) is marked as **suppressed**.
   - Its posting list is cleared from RAM.
   - Prevents memory blowup from uninformative high-frequency terms.
   - All suppressed keys and counts are logged in `blocking_stats.json`.
3. **Compact Metadata Lookup**: Target source tags (`Source 2` vs `Source 3`) and canonical country codes are stored in minimal string dictionaries.
4. **Deterministic Ordering**: All candidate generation and index traversals produce deterministically ordered results (sorted by `target_entity_id`).

---

## 5. Candidate Generation & Filtering Engine

For each Source 1 query entity:
1. Generate blocking keys via configured strategy or composite.
2. Query `BlockingIndex` (skipping suppressed keys).
3. Deduplicate target IDs across strategies.
4. Apply country agreement policy:
   - `allow_missing` (default): Drops candidates only if both S1 and target have non-empty countries that explicitly conflict (`country_s1 != country_target`). Preserves matches when either entity has an empty or unknown country.
   - `strict`: Requires exact country agreement; drops matches if either country is missing.
   - `none`: Does not filter on country.
5. Record source identity (`Source 2` or `Source 3`).
6. Record generating strategy names.
7. Sort candidate pairs deterministically.

---

## 6. Evaluation Methodology & Tradeoff Metrics

Candidate generation is evaluated against training ground truth (`train_ground_truth.tsv`):

### Core Formulas:
1. **Candidate Recall**:
   $$\text{Recall} = \frac{\text{Recovered True Positive Pairs}}{\text{Total Authoritative Ground Truth Pairs}}$$
2. **Candidate Reduction Ratio**:
   $$\text{Reduction Ratio} = 1.0 - \frac{\text{Total Generated Candidate Pairs}}{|S1| \times (|S2| + |S3|)}$$
3. **Distribution of Candidates per S1**: Mean, Median, Min, Max, P90, P95, P99.
4. **Per-Source Recall**: Breakdown for Source 2 matches vs. Source 3 matches.
5. **Per-Country Recall**: Breakdown across all represented countries (US, IN, FR, DE, etc.).
6. **Zero-Match Behavior**: Volume of candidates generated for singletons (S1 with 0 ground truth matches).
7. **Multi-Match Behavior**: Recovery rate for S1 entities with $\ge 2$ target matches.
8. **False Negative Diagnostics**: Detailed logs of missed pairs with candidate-level diagnostic attribution.

---

## 7. Phase Boundaries

| Dimension | Phase 4: Blocking & Filtering | Phase 5: Final Candidate Pipeline | Phase 6: Feature Engineering |
| :--- | :--- | :--- | :--- |
| **Objective** | Explore, benchmark & evaluate blocking strategies | Final locked candidate generation pipeline for modeling | Pairwise similarity feature extraction |
| **Output File** | Experimental `candidate_pairs.tsv` + benchmark stats | Production `candidate_pairs.tsv` | `features.parquet` / tabular feature matrices |
| **Fuzzy Matching** | **None** (index lookup only) | None | Expensive token, Levenshtein, Jaro-Winkler, embeddings |
| **Scope** | Optimization of recall vs. reduction frontier | Scaled execution across full dataset splits | Model training and inference data prep |

---

## 8. CLI Usage Guide

### A. Local Prototyping & Sample Evaluation
Run lightweight evaluation on synthetic fixtures or local samples:

```bash
# Evaluate all strategies and benchmark recall on local sample:
python scripts/evaluate_blocking.py \
    --raw-input ./tests/fixtures/sample_data \
    --norm-input ./tests/fixtures/sample_data \
    --output ./artifacts/candidates/block_v001 \
    --sample-size 1000 \
    --seed 42

# Generate candidate pairs directly:
python scripts/generate_candidates.py \
    --norm-input ./tests/fixtures/sample_data \
    --output ./artifacts/candidates/block_v001 \
    --strategy composite
```

### B. SageMaker / S3 Execution
On AWS compute targeting the full dataset in S3:

```bash
# Full strategy evaluation benchmark in SageMaker:
python scripts/evaluate_blocking.py \
    --raw-input s3://sagemaker-ap-southeast-2-904290466033/raw/dataset/ \
    --norm-input s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
    --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/ \
    --aws-region ap-southeast-2 \
    --sample-size 10000 \
    --max-posting-list-size 500

# Full candidate generation for Phase 5 preparation:
python scripts/generate_candidates.py \
    --norm-input s3://sagemaker-ap-southeast-2-904290466033/artifacts/normalized/v001/ \
    --output s3://sagemaker-ap-southeast-2-904290466033/artifacts/candidates/block_v001/ \
    --aws-region ap-southeast-2 \
    --strategy composite \
    --max-posting-list-size 500
```

---

## 9. Output Artifacts

Phase 4 generates 5 structured artifacts in the output destination:
1. `candidate_pairs.tsv`: Generated candidate pairs with schema `source1_entity_id`, `target_entity_id`, `target_source`, `strategies`.
2. `blocking_stats.json`: Primary metrics (recall, candidate count, reduction ratio, candidate distribution percentiles, per-source/country breakdown).
3. `blocking_strategy_results.json`: Full benchmark comparison across all evaluated strategies with false negative error analysis.
4. `blocking_validation_report.json`: Phase 4 validator report verifying the 12 integrity rules.
5. `blocking_metadata.json`: Provenance metadata linking commit hash, input normalization version (`v001`), timestamp, and configuration parameters.
