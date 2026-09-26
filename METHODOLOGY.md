# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source 2 / Source 3 business records to Source 1 entities with a CPU-only, three-step pipeline:
1. **Candidate generation:** a sparse TF-IDF top-K search over phonetic and exact-spelling name tokens and address tokens.
2. **Scoring:** two LightGBM models. The first scores each candidate pair; the second re-scores it in the context of the record's other candidates.
3. **Decision:** an entity-level rule that picks, for every Source 1 entity, the set of matches with the highest expected F0.5.

The main ideas behind the gains are:
- a stage-2 model trained on out-of-sample stage-1 scores, from three disjoint held-out entity sets;
- a decision rule that optimises the macro F0.5 metric directly, rather than using one global threshold;
- making the pipeline robust to the test set's higher density of look-alike records.

Validation macro F0.5 is **0.9781** on the held-out entities, and **0.9755** on a check that simulates the test set's density of look-alike records.

---

## 2. Methodology

### 2.1 Problem Analysis

**Data scale.**

| Split | Source 1 entities | Source 2 + 3 records |
|---|---|---|
| Train | 2.21 M | 10.3 M |
| Test | 1.73 M | 9.97 M |

**Match structure.** Every Source 2/3 record in the training labels belongs to at most one Source 1 entity. An entity has 0–5 matches from Source 2 and 0–6 from Source 3, about 3.3 in total on average. About 6% of entities are singletons.

**Duplicated names.** About 50% of Source 1 entities share their normalised name with another Source 1 entity: "Shivam Trading", "Om Services Private Limited", and so on. Names alone are therefore not enough, and address and house-number evidence decides many matches.

**Noise patterns.**
- **Names:**
  - legal-suffix variants (Pvt Ltd / Private Limited / (Ltd) / [LLC], SARL / SAS / EURL);
  - suffixes added or dropped;
  - word order swaps and typos ("Glbsal", "Pulbldic", "5tar");
  - joined-up web-style names ("pramabrothers.com");
  - tags like "(India)" or "(France)";
  - **native-script names** (Devanagari, Kannada, Bengali, Tamil and others) for about a quarter of Indian query records.
- **Addresses:**
  - components reordered, with the state first or last;
  - abbreviations (St/Street, Rd, Av., Bd., R. = Rue);
  - states given as codes or names ("MH" / "Maharashtra" / "महाराष्ट्र");
  - missing components;
  - landmark-style addresses ("Opp. Nabard, Nr. Usmanpura Garden");
  - about 4% of records have **no address at all**.

**Train/test shift.**
- **New country:** test adds France (15% of Source 1 entities), which never appears in training.
- **Crowding:** test has about 5.8 Source 2/3 records per Source 1 entity against 4.7 in train, so there are roughly twice as many look-alike non-matching records ("confusers") per entity.
- **Country mix:** test is 47% India (train 40%), and India is the harder country (validation F0.5 0.971 vs 0.980 for US).

### 2.2 Solution Strategy

**Approach type:** blocking + two-stage gradient-boosted classifier + metric-aware decision rule.

**Core innovation:**
- **Metric-aware decisions:** the final matches maximise expected per-entity F0.5, not a global threshold.
- **Out-of-sample stage 2:** the stage-2 re-scorer is trained on out-of-sample stage-1 scores.
- **Robustness to crowding:** entity-side context features are dropped, and a prior-shift correction is tuned on a validation set that simulates test's density of look-alike records.

**The pipeline:**
1. **Normalisation** (`normalize.py`, `prep.py`):
   - transliterate every script to ASCII (anyascii);
   - strip legal suffixes, stop words and honorifics into a *name core*;
   - build a phonetic *skeleton* (vowels dropped, similar consonants merged, so "praivet limitet" ≈ "private limited");
   - expand address abbreviations;
   - canonicalise states (US states, Indian states and their spellings, French regions and departments);
   - extract house and street numbers.
2. **Candidate generation** (`blocking.py`): see §3.
3. **Stage 1** (`train.py`): a LightGBM pair classifier with 66 features.
4. **Stage 2** (`train_s2.py`): a LightGBM re-scorer with 43 features, trained with 5-fold CV on held-out entities.
5. **Decision rule** (`decide.py`): expected-F0.5 optimisation per entity.

---

## 3. Candidate Generation (Blocking)

**Method.** Every record becomes an IDF-weighted sparse bag of tokens, prefixed with its country. For each Source 2/3 record (query), the Source 1 records with the highest cosine similarity are retrieved with `sparse_dot_topn` (multithreaded sparse top-K matrix product). The candidate set is the union of two indexes:
- **name + address index:** top K = 20;
- **name-only index:** top K = 10. It catches records whose address is missing or unhelpful.

**Blocking keys (token types):**

| Token | Example | Purpose |
|---|---|---|
| name skeleton word | `n|prvt` | transliteration- and typo-tolerant name match |
| skeleton word pair | `m|svn_trtng` | word order and phrase evidence |
| despaced skeleton | `n|svntrtng` | "colonialfoods.com" vs "Colonial Foods" |
| **exact name word** | `r|sweven` | separates distinct names that share a skeleton |
| **exact word pair** | `s|sweven_trading` | the same, for pairs |
| address word | `a|thane`, `a|#105` | location evidence |
| adjacent address pair | `b|#105_saiba` | house number + street |

Tokens found in more than 3,000 Source 1 records are dropped from the index to keep the product sparse.

**Candidate pairs generated:**
- **train:** 254 M (24.6 per query);
- **test:** 248 M (24.9 per query).

**Keeping true matches.**
- **Recall:** on a 300k-query training sample, 97.68% of true pairs are among the candidates.
- **Two indexes:** the name-only index covers records without a usable address.
- **Exact-spelling tokens:** an error analysis showed the phonetic skeleton merged distinct names ("Sweven", "Seven" and "Shivam" all become `svn`). A unique business could then be crowded out of the top K by dozens of same-skeleton entities. Adding exact name words and word pairs cut blocking misses by 20% (recall 97.10% → 97.68%) with 5% *fewer* candidate pairs, and raised stage-1 validation F0.5 from 0.9673 to 0.9692.
- **Blocking scores as features:** the similarity score and rank of each candidate within its query are passed to the classifier. They are its most informative features.

---

## 4. Matching Model

### Stage 1: pair classifier (LightGBM, 66 features)

- **Name features:**
  - IDF-weighted cosine and query/entity coverage over name tokens;
  - RapidFuzz ratio, token-sort, token-set and partial ratios, Jaro-Winkler;
  - despaced-name match, skeleton ratio and token-set, alternative (aka/dba) name match;
  - lengths and token counts.
- **Address features:**
  - IDF cosine and coverage over address tokens;
  - RapidFuzz ratio, token-set and token-sort on the address;
  - state equality (−1 / 0 / 1 for mismatch / unknown / match).
- **Number features:**
  - cosine and coverage over number tokens;
  - shared numbers, first-number equality or containment;
  - numbers present only on the query side.
- **Other features:**
  - legal-form equality;
  - query has no address;
  - query name is in a native script;
  - query source (S2/S3), number of candidates, blocking scores and ranks.
- **Within-query features:** for the key similarities, the gap to the best candidate of the same query and the rank within the query. Each query can match at most one entity, so its competitors are informative.
- **Training:** 500k training queries (about 13 M labelled pairs). Settings: 255 leaves, learning rate 0.05, early stopping on the validation queries.

### Stage 2: context re-scorer (LightGBM, 43 features)

- **Inputs:**
  - the stage-1 probability and its query context: gap to the query's best, maximum, sum, number of candidates above 0.1, rank;
  - key stage-1 similarities;
  - fine-grained features (`fine.py`): the IDF mass and length of name words not shared by the two sides (worst unmatched word on each side), legal words present on only one side, title-word differences, house-number equality and numeric distance, numbers present only on the entity side.
- **Training data:** only entities whose stage-1 scores are out-of-sample. That means the stage-1 validation entities plus two further disjoint held-out sets (8% and 10% of Source 1 entities, `extend_val.py`), about 0.44 M entities in total. Training uses 5-fold cross-validation grouped by entity.
- **Robust to crowding:** entity-side context features, which change when an entity has more look-alike records competing for it, are **excluded** (`--noent`). Because the remaining features are all query-side, the effect of test-like crowding can be simulated exactly: non-matching query rows are duplicated and the decision rule is re-scored.
- **Prior-shift correction:** probabilities are rescaled by an odds multiplier r (p' = rp / (rp + 1 − p)). r is chosen on this test-like simulation, which gives r = 0.5.

### Decision rule (threshold selection)

1. Each Source 2/3 record keeps only its best Source 1 candidate.
2. For each Source 1 entity, the records that chose it are sorted by probability.
3. The entity keeps the top k (k = 0 allowed) that maximise **expected** F0.5, estimated by sampling outcomes from the calibrated probabilities.

This directly optimises the macro metric. A singleton entity drops from F = 1 to 0 on one false merge, while a large entity can afford a doubtful addition. On plain validation it is within ±0.0005 of the best global threshold. Its advantage is that it needs no threshold tuned on the training distribution, and it adapts per entity once the prior shift is applied.

**Model type:** LightGBM gradient-boosted decision trees (MIT license). Both models are trained from scratch on the provided data. No pretrained models, external data or APIs are used.

---

## 5. Results & Error Analysis

| Configuration | Validation macro F0.5 |
|---|---|
| Stage 1 only (global threshold) | 0.9692 |
| + Stage 2 (global threshold) | 0.9783 |
| + Stage 2, expected-F rule | **0.9781** |
| Test-like simulation (2× confusers), expected-F rule, shift 0.5 | 0.9755 |

**Leaderboard (public):** 0.968 for the earlier versions. Final version: [fill in].

**Where the remaining error is** (stage-1 validation; the gain is the F0.5 recovered if a whole segment were fixed):

| Segment | Share of true pairs | Correctly matched | Upper-bound gain |
|---|---|---|---|
| Query with no address | 4.4% | 45% | +0.008 |
| Native-script name (India) | 9.5% | 91% (vs 93% for Latin script) | +0.005 |
| True match not among candidates | 2.3% | 0% | ~+0.008 (estimated) |

The first three rows of the results table are on the original 2% held-out entities (44,136 Source 1 entities).

**Common false positives (wrong merges):**
- Same-name businesses at the same or nearby addresses with a different trade word ("VA Club" vs "VA Union SAS" at 5 Rue de Tunis).
- Chains and franchises that share a name across cities when the query has no address.
- A different legal entity of the same group ("Linfo & Cie SCI" vs "Linfo & Cie SA").

The precision-weighted metric and the expected-F rule make the model decline these when the evidence is ambivalent.

**Common false negatives (missed matches):**
- Queries without an address whose name is shared by many entities. These are often irreducibly ambiguous.
- Heavily transliterated native-script names ("ಬಾಂಬೆ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್" → "baambe kansstrkshns" vs "Bombay Constructions").
- Multi-character typos in short names.
- Addresses that differ in every component apart from the city.

---

## 6. Conclusion

A carefully engineered sparse candidate search plus two gradient-boosted models reaches about 0.978 validation macro F0.5 on CPU only. The largest single gain came from the stage-2 context re-scorer trained on out-of-sample stage-1 scores (+0.009 over stage 1).

The main lessons:
- The phonetic keys that make transliterated names match also merge distinct names. Pairing them with exact-spelling keys recovers recall at no cost.
- A train/test shift in how many look-alike records compete for each entity is best handled with features that stay valid under it, plus an explicit prior-shift correction.
- The remaining error is dominated by records without an address and by native-script names. A multilingual text encoder is the natural next step there.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains:
- `src/`: all pipeline code;
- `README.md`: setup, run instructions and file roles;
- `requirements.txt`: pinned Python 3.12 dependencies;
- `run_all.sh`: runs the whole pipeline.

Set `ER_DATA_DIR` to the dataset folder, then run:

```bash
bash run_all.sh
```

It produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`:

| Step | Command | Time (10 threads) |
|---|---|---|
| Normalise | `prep.py train test` | ~5 min |
| Candidates | `blocking.py train`, `blocking.py test` | ~15–18 min each |
| Stage 1 | `train.py 500000` | ~22 min |
| Extra held-out scores | `extend_val.py 0.08`, `extend_val.py 0.10 --out=val3` | ~20 min each |
| Stage 2 | `train_s2.py --ext=3 --noent --tag=_noent3` | ~23 min |
| Test inference | `predict.py --tag=_noent3` | ~3 h |

The hardware used had 16 GB RAM and no GPU, so stages must run one at a time.

### B. Additional Results

**Candidate generation, 300k training queries:**

| Blocking variant | Pairs per query | Recall |
|---|---|---|
| Phonetic + address tokens | 25.9 | 0.9710 |
| + exact-spelling name tokens (final) | 24.6 | 0.9768 |

**Stage 2 across versions** (test-like simulation, best shift):

| Version | F0.5 |
|---|---|
| Entity features, 2 held-out sets | 0.9728 |
| No entity features, 2 sets | 0.9738 |
| No entity features, 3 sets | 0.9739 |
| + exact-spelling blocking (final) | **0.9755** |
