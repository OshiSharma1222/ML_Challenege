# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

A CPU-only, three-stage pipeline.
- **Candidate generation:** every record is transliterated to ASCII and reduced to phonetic, vowel-insensitive tokens. Sparse TF-IDF top-K search then produces candidates from two indexes (name+address, and name-only).
- **Stage 1:** a LightGBM model scores ~263M test candidate pairs using 68 country-agnostic similarity features.
- **Stage 2:** a second LightGBM re-scores the surviving pairs, adding context (competing candidates of the same record, other records pointing at the same entity) and fine-grained features aimed at deliberate "confuser" records.

The key structural insight is that every Source 2/3 record belongs to at most one Source 1 entity. Matching is therefore an assignment problem: each record goes to its best entity, or to none.

Validation macro F0.5 (held-out Source 1 entities, singletons included): **0.9753**.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M Source 1 and 10.3M Source 2/3 training records:

- **Assignment structure.** 7.64M Source 2/3 records are matched, and each appears in exactly one ground-truth list. 2.68M records (26%) match nothing. On average an S1 entity has 3.46 matches (0–11); only 5.6% are singletons.
- **Scripts and transliteration.** Many Indian Source 2/3 names are written in Devanagari, Tamil, Bengali, Kannada, etc. (e.g. `राम मार्केटिंग प्राइवेट लिमिटेड` = "Ram Marketing Private Limited"). State names also appear in native script (`हरियाणा`, `ಕರ್ನಾಟಕ`, `পশ্চিমবঙ্গ`) or as codes (`MH`, `WB`, `OH`).
- **Name noise.** Legal suffixes are added, dropped, moved to the front, or bracketed (`LLC Sta1der Ínterprivate`, `*** RASA PURE [PRÍVATE-LIMITED]`). Other noise includes words reordered, typos and OCR errors (`8ig 64 Grill`), accents injected (`Cáyman`), domain forms (`colonialfoods.com`), `aka` trade names, and honorific prefixes (`Dr`, `Smt`, `Shri`).
- **Address noise.** Components are reordered (`AZ, 608 THIRD STREET N, CLARKDALE`), abbreviated (St/Street, Rd, Bd, R.), dropped (no street, no house number), or given as landmarks (`Near Bus Stand`). House numbers get zero-padded (`00409`) or appear as ranges (`419-423`). About 3.4% of Source 2/3 addresses are missing entirely.
- **Confusers (key finding from error analysis).** The data contains unmatched records built to look like a real entity: the same address with one name word swapped (`Riley Allied Tungsten Corp` vs `Elliott Allied Tungsten Corp`), a nearby house number (8113 vs 8112), a different legal form (LLP vs Pvt Ltd), or a different professional title (DLS vs DDS). These cause most false merges.
- **France** is present only in the test set (~260k S1 entities), so no country-specific logic or country one-hot is used anywhere.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier, with record-to-entity assignment
**Core Innovation:**
1. Phonetic consonant "skeleton" keys, which make transliterated Indic names and English spellings collide: `praivet limitet` → `prvt lnt` ← `private limited`.
2. Treating matching as an assignment problem, and giving the model within-record competition features (the margin over the best competing entity).
3. A stage-2 re-scorer with entity-level context and word-alignment / house-number features that target confuser records.

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`normalize.py`):
- anyascii transliteration of all scripts, then lower-casing and removal of punctuation.
- `&`→`and`; web domains split out.
- Legal suffixes are separated from the "core" name, including transliterated forms matched via their skeleton; stop words and honorifics are removed.
- Address abbreviations are expanded; ordinals and zero-padding are normalised.
- Each comma component is checked against US/Indian state names, codes and native-script spellings, and mapped to a canonical state token.

**Skeleton**: `ph→f, sh→s, ch/c/q/ck→k, x→ks, z→s, w→v, j→g, d→t, m→n`, then vowels, `y` and `h` are dropped and repeats collapsed.

**Blocking tokens** (all prefixed with the record's country, an open set):
- name skeleton words and name bigrams;
- the despaced full-name skeleton (catches `colonialfoods.com`);
- the `aka` alternative name;
- address word skeletons, house/unit numbers, and address bigrams (e.g. `17560_ls`).

**Search:**
- IDF weights are computed on Source 1; tokens with df > 3000 are dropped; vectors are L2-normalised.
- For every Source 2/3 record, `sparse_dot_topn` (multithreaded sparse matrix product) returns the top 20 S1 entities from the name+address index and the top 10 from a name-only index. The name-only index recovers records with no address.
- The union has about 26 candidates per record.

**Recall** (fraction of true record→entity links present among candidates, 300k-record sample):

| K | full index | name index | union |
|---|---|---|---|
| 1 | 0.913 | 0.348 | 0.917 |
| 5 | 0.956 | 0.520 | 0.958 |
| 10 | 0.965 | 0.572 | 0.967 |
| 20 | 0.972 | – | **0.973** |

- **Candidate pairs generated:**
  - blocking: 267.5M (train) and 262.7M (test), about 26 per record, versus 2.2M × 10.3M ≈ 2.3×10¹³ possible pairs;
  - after the stage-1 learned filter (p ≥ 0.001), about 1.2 pairs per record remain. These are the pairs the final model runs on, and they are what `candidate_pairs.tsv` lists.
- **How true matches were not lost:**
  - two complementary indexes;
  - phonetic keys robust to transliteration and vowel typos;
  - bigram tokens that keep specificity after the common-token cap;
  - the stage-1 pruning threshold was chosen so it retains 99.98% of true pairs.
  - Most remaining blocking misses are records with no address and a generic name that many S1 entities share. Under F0.5, these could not be merged safely anyway.

---

## 4. Matching Model

**Stage 1 — pair model** (`features.py`, 68 features, all computed for every candidate pair):
- **Name features:**
  - rapidfuzz ratio / token_sort / token_set / partial ratio and Jaro-Winkler on the core name;
  - ratio and token_set on the phonetic skeleton;
  - ratio and partial ratio on the despaced name;
  - token_sort on the full name, including legal words;
  - best `aka`-alternative token_set;
  - IDF cosine, query coverage and entity coverage of the name tokens;
  - name lengths and token counts;
  - legal-form agreement (+1 / 0 / −1);
  - whether the query name is in a non-Latin script.
- **Address features:**
  - rapidfuzz ratio / token_set / token_sort / partial_token_set;
  - IDF cosine and coverage of the address word/bigram view and of the number view;
  - house numbers: counts, shared count, first-number equality and membership, query-only numbers;
  - state agreement (+1 / 0 / −1);
  - missing-address flag and lengths.
- **Blocking and competition features:**
  - blocking scores and ranks from both indexes;
  - number of candidates;
  - for 7 key similarities, the margin over the best competing S1 candidate of the same record, and the rank within that record.

**Stage 2 — context re-scorer** (`stage2.py`, `fine.py`), run on pairs with stage-1 p ≥ 0.001:
- **Record context:** stage-1 p; margin to the record's next-best entity; the record's max and sum of p; number of candidates with p > 0.1; rank.
- **Entity context:** how many *other* records confidently pick this entity; the entity's sum and max of p; this record's rank among the entity's records.
- **Fine-grained features:**
  - word-level name alignment in both directions: the number of words with no counterpart (fuzzy ratio < 75 and different skeleton), their summed IDF and IDF fraction, the longest unmatched word, and the worst word match;
  - whether the first words agree;
  - legal tokens present on only one side, including transliterated `pra li`;
  - professional-title disagreement (MD/DO/DDS/DLS…);
  - house number: exact first-number equality, log absolute differences, entity-only numbers.

**Model type:** LightGBM binary classifiers (MIT license, trained from scratch; no pretrained or external model).
- Stage 1: 255 leaves, lr 0.05, early stopping, trained on all ~13M candidate pairs of 500k training records.
- Stage 2: 63 leaves, 600 rounds, 5-fold CV grouped by true entity.

**Decision rule:** each Source 2/3 record is assigned to its highest-probability S1 candidate if p ≥ threshold, otherwise to no entity. An S1 entity's match list is the set of records assigned to it; an empty list marks it as a singleton.

**Threshold selection method:** a sweep of macro F0.5 on held-out entities; best threshold 0.75. The stage-2 CV curve is flat between 0.60 and 0.80 (±0.0002).

---

## 5. Results & Error Analysis

**Validation protocol:**
- 2% of Source 1 entities (44,136) are held out, and none of their true-match records is used in training.
- Every record that truly matches a held-out entity, or has one among its top blocking candidates (520k records), is scored against **all** of its candidates.
- Those records are then assigned, and macro F0.5 is computed over the held-out entities exactly as the leaderboard does, singletons included.

| Model | Macro F0.5 |
|---|---|
| Stage 1 (pair features) | 0.9673 |
| + Stage 2 context features | 0.9697 |
| + Stage 2 fine-grained features | **0.9753** |

Stage-1 breakdown (threshold 0.65):
- non-singleton entities: mean precision 0.985, mean recall 0.932;
- singleton entities: F0.5 ≈ 0.95;
- of the true record links: 93.2% assigned correctly, 0.14% assigned to a wrong entity, 6.7% left unassigned.

- **Common false positives (wrong merges):** mostly (88%) unmatched confuser records rather than records confused between two real entities. Typical forms:
  - the same address with one distinctive name word replaced;
  - a house number off by a few;
  - an `LLP` versus `Pvt Ltd` / `प्रा. लि.` variant;
  - an appended `Co`;
  - a changed professional title.

  The stage-2 fine features target exactly these.
- **Common false negatives (missed matches):**
  - records with no address whose generic name is shared by many S1 entities;
  - Indic-script names whose transliteration drifts (`teknolojij` vs `technologies`) combined with a truncated address;
  - heavy address truncation combined with name word replacement.

  Under F0.5, leaving these unassigned is the better trade-off.

---

## 6. Conclusion

The biggest gains came from three things:
1. recognising the one-entity-per-record assignment structure;
2. transliteration-aware phonetic blocking, which reaches 97.3% candidate recall with about 26 candidates per record;
3. modelling competition between candidates.

A small second-stage model aimed at the confuser records found in error analysis added another +0.8 F0.5 points. Everything is country-agnostic, so France is handled by the same features and model.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `README.md` (exact commands), `requirements.txt` (pinned), `run_all.sh` (end-to-end);
- `src/` — `prep.py` → `blocking.py` → `train.py` → `train_s2.py` → `predict.py`.

`predict.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Hardware used: 18-core CPU, 16 GB RAM, no GPU. Total runtime is about 2.5 h.

### B. Additional Results

Top stage-1 features by gain:
- `bs_f_gap` (blocking-score margin over the best competing entity);
- `bs_f_rk`;
- `a_tset_gap`;
- `num_qonly` (query house numbers absent from the entity);
- `cove_num`;
- `n_ratio_gap`;
- `br_f`;
- `full_tsort`;
- `bs_f`;
- `n_tset_gap`.
