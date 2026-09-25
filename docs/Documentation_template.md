# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]  
**Team Members:** [Gaurav Goyal, MEMBER 2, MEMBER 3, MEMBER 4]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source-1 entities against Sources 2 and 3 with a three-stage pipeline: (1) a
country-partitioned, multi-view candidate generator that unions six sparse TF-IDF views (character
n-grams and word tokens over normalised names, addresses and a phonetic key) [+ a MiniLM dense
view on GPU], reaching a pair recall of ~98.6% at ~64 candidates per entity; (2) a gradient-boosted
pairwise matcher over 91 similarity, context and uniqueness features, trained with group-wise
cross-validation on the provided labels; (3) a precision-first decision rule (global threshold tuned
for macro-F0.5 on out-of-fold predictions, plus a one-to-one constraint that follows from Source 1
being deduplicated). Two innovations mattered most: a rule-based Indic-script transliterator with a
phonetic key that bridges the ~24% of Indian vendor records written in Devanagari/Tamil/Kannada/…,
and per-country *name/address uniqueness counts* that let the model resolve name-only (empty
address) and trade-name records. [Final validation macro-F0.5: X.XXXX; public LB: X.XXXX.]

---

## 2. Methodology

### 2.1 Problem Analysis

Key facts from EDA on the training data (2.21M S1 / 5.03M S2 / 5.29M S3 records):

- **Match cardinality is high.** Only 5.6% of S1 entities are singletons; the mean entity has 3.46
  matches (max 11), roughly half from each vendor source. Each S2/S3 record belongs to at most one
  S1 entity (a strict 1-to-1 property we verified on all 7.6M labelled pairs), and every labelled
  pair is within the same country. Recall therefore matters more than the "precision-weighted"
  metric suggests: an all-empty submission scores 0.056.
- **Scripts.** About 24% of Indian S2 names (and ~12% of S3 names) are phonetic transliterations of
  the English name into an Indic script (Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada,
  Malayalam, Odia), e.g. "Digital Constructions Private Limited" ↔ ಡಿಜಿಟಲ್ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್.
  State names in addresses are also sometimes in Indic script.
- **Name noise.** ALL-CAPS (28% of S2), legal-form variants (Pvt/Private/प्रा. लि., L.L.C./LLC,
  Inc/Incorporated), "&" vs "and", leet-style typos (Precisi0n, 5ervices), scrambled words
  (Deintel/Dombcla for Dental), accented characters (Déntal), domain names used as names
  (hightowerarray.com ↔ Hightower Array Co), "The" prefixes, DBA constructs ("X dba Y") and —
  hardest — fully replaced trade names at the same address (a single invented token such as
  "Vantageonyxnyla" labelled as a match).
- **Address noise.** Abbreviations (Rd/Road, Ave/Avenue, Blvd), state names vs codes (Illinois/IL,
  Maharashtra/MH, truncations like "maharashtr"), house-number labels (H No, Door No, Plot No, #),
  dropped or altered digits, component reordering (state or city first), missing components,
  landmark phrases ("Near SBI ATM", "Opp. Bus Stand"), literal "null" tokens, and 3.8% empty
  addresses in S2/S3. Look-alike non-matches are typically the same name at a different address, a
  different real-word name at the same address, or a small house-number edit.
- **Open-set countries.** Training covers US and India; the test set adds France (15% of test S1).
  All normalisation is therefore country-agnostic (dictionaries cover US/IN/FR forms but nothing is
  filtered by country), and the country label is used only as a blocking partition.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier (hybrid sparse [+ dense] candidate generation,
gradient-boosted matcher, F0.5-tuned decision rule with a one-to-one constraint).  
**Core Innovation:** (a) cross-script name matching via a rule-based Indic→Latin transliterator (one
offset table for nine Unicode blocks) and a consonant-skeleton phonetic key used both as a blocking
view and as features; (b) per-country uniqueness counts (how many S1 entities share a candidate's
core name / phonetic key / no-space name, how many records share its address) that turn ambiguous
name-only records into resolvable ones; (c) a fully memory-lean, per-country streamed
implementation that runs on a shared 15 GB machine.

Pipeline overview:

```
raw TSVs ──► normalise (translit, fold, legal forms, address canon, state codes, phonetic key)
          ──► per-country uniqueness counts
          ──► blocking: per country, per 100K-S1 block, top-k (k=10) per view per source, union
          ──► stage A features (cosines, ranks, gaps, margins, reverse ranks, mutual-best)
          ──► pruning (score threshold calibrated to keep 99.8% of training positives)
          ──► stage B features (string, token, phonetic, legal, number, uniqueness, flags)
          ──► GBDT matcher (GroupKFold OOF by S1 entity)
          ──► decision: global threshold (OOF-tuned) + one-to-one filter ──► matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

Blocking runs **within country** (all labelled matches are same-country) and within blocks of
100K S1 entities, against all S2 and all S3 records of that country. For every S1 entity and each
source we retrieve the top-k (k = 10) records under each of six L2-normalised TF-IDF views and take
the union:

| view | text | analyser | what it catches |
|---|---|---|---|
| name_c3 | core name (legal forms stripped) | char 3-grams | typos, spacing, punctuation |
| name_w | core name | word tokens | rare distinctive words |
| name_ph | phonetic key of the core name | char 3-grams | cross-script transliterations, spelling variants |
| addr_c3 | normalised address | char 3-grams | same building, abbreviation variants |
| addr_w | normalised address | word tokens | reordered / partial addresses |
| full_w | core name + address | word tokens | chains, partial overlap on both fields |
| [dense] | "name, address" MiniLM-L6-v2 embedding (Apache-2.0, 22M params) | exact inner-product top-k on GPU | semantic / heavy-noise variants |

Vectorisers use `max_df = 0.01` (n-grams present in >1% of records are dropped), which keeps the
sparse products small and, empirically, costs no recall. Top-k is computed with
`sparse_dot_topn` (chunked scipy fallback otherwise); the dense view uses exact search in
memory-sized chunks rather than an approximate index.

- **Blocking keys used:** the six TF-IDF views above (+ dense), country partition.
- **Candidate pairs generated:** [N total; ~82 per S1 before pruning, ~64 after] — the pruned set is
  exactly `candidate_pairs.tsv`.
- **How we ensured true matches were not lost:** measured pair recall on the training labels for
  every view and k (recall-vs-k curves): the union reaches 98.6% pair recall at k = 10 (US 99.4%,
  India 97.5%) versus 97.1% for the four "classic" views alone; the pruning threshold is calibrated
  on training positives to keep 99.8% of them. The single strongest view is `full_w` (94.4% India /
  98.9% US alone); `name_ph` adds the transliterated records that no other view can reach.

---

## 4. Matching Model

**Features used (91):**
- Name features: TF-IDF cosines per view; Jaro-Winkler on core name, full name, phonetic key,
  no-space name; token Jaccard/containment; exact/containment/first-token equality; acronym match;
  length ratio; extra tokens on either side and how many of them are non-generic words; legal-form
  equality / conflict / missing; Indic-script flags.
- Address features: address char-3-gram and word cosines; Jaro-Winkler on the normalised address;
  house/PIN number Jaccard, any/both/first-number equality, Jaro-Winkler of the first numbers,
  numbers present on one side only; ZIP/PIN equality and conflict; landmark ("near/opp") flags;
  address empty flag and lengths; country equality.
- Context / uniqueness: rank and gap to the best candidate per (S1, source) group for every view and
  for the overall score; margin between the top two; number of close candidates; the candidate's
  reverse rank among competing S1 entities and its competitor count; mutual-best flag; per-country
  counts of S1 entities sharing the S1's / candidate's core name, phonetic key and no-space name;
  counts of records sharing the address; number of retrieving views.

**Model type:** LightGBM gradient-boosted trees (700 trees, 63 leaves, lr 0.05) [sklearn
HistGradientBoosting fallback], fitted on all positives plus a 50% weighted sample of negatives;
5-fold GroupKFold by S1 entity for out-of-fold probabilities.

**Threshold selection method:** macro-F0.5 maximised on out-of-fold predictions over a grid of global
thresholds (chosen ≈ 0.80) combined with a one-to-one filter (each S2/S3 record is assigned only to
the S1 entity with the highest probability, since Source 1 is deduplicated). Alternatives evaluated
on OOF and rejected because they did not improve F0.5: per-entity expected-F0.5 set selection,
isotonic calibration, per-country thresholds, a second-stage model over group-level probability
features, and a second-stage model over candidate–candidate similarities.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [validation: X.XXXX (OOF, N S1 entities); held-out train-derived slice:
  X.XXXX; public leaderboard: X.XXXX]. Development trajectory on an 8% slice (40K training
  entities): 0.966 → 0.970 (word views) → 0.977 (uniqueness/extra-token/number features) OOF; holdout
  0.974 → 0.977 → 0.983.
- **Common false positives (wrong merges):** look-alike records that share the address but carry a
  different real-word name ("Vijayawada Taxi Center" vs "Vijayawada Power Private"); the same name
  with an added or changed word ("Rowe and Kister" vs "Rowe and Kister Holdings"); small
  house-number edits (67-21 vs 67-23) that the labels treat as different premises; singleton S1
  entities with a near-identical look-alike (about 5% of singletons).
- **Common false negatives (missed matches):** (i) name-only S2/S3 records with an empty address
  whose name is shared by several S1 entities — inherently ambiguous, and correctly left unmatched
  under F0.5 when the uniqueness count is > 1; (ii) fully replaced trade names at a perturbed
  address; (iii) heavily scrambled names combined with a missing address; (iv) residual blocking
  misses (~1.4% of pairs), concentrated in Indic-script names whose transliteration differs
  strongly from the English spelling.
- Per-country behaviour: [predicted-empty rate and mean matches per country for US / India / France;
  France is unseen in training and is monitored through these statistics].

---

## 6. Conclusion

A carefully normalised, multi-view sparse blocking stage with a phonetic/transliteration view gives
a high recall ceiling at a small candidate budget; a gradient-boosted matcher over similarity,
context and uniqueness features then delivers precision, and a simple OOF-tuned threshold with a
one-to-one constraint converts probabilities into F0.5-optimal match sets. The main lessons: recall
ceilings and data-specific normalisation (scripts, legal forms, state codes, number labels) moved the
score far more than any post-hoc decision machinery, and a per-country streamed implementation was
necessary to process 24M records on modest shared hardware.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — run from that folder:

```
pip install -r requirements.txt
python -m src.pipeline --data-dir data --out-dir output --n-jobs 8 --max-df 0.01 --train-s1 150000 [--dense]
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

- `src/translit.py` — Indic→Latin transliteration and phonetic key
- `src/normalize.py` — folding, legal forms, address canonicalisation, state codes, number labels
- `src/store.py`, `src/data.py` — per-country columnar caches, loading
- `src/blocking.py`, `src/dense.py` — sparse and dense candidate generation
- `src/features.py` — stage A / stage B features
- `src/model.py` — GBDT, GroupKFold OOF, decision rules
- `src/pipeline.py` — end-to-end orchestration writing `output/matching_results.tsv` and
  `output/candidate_pairs.tsv`; `src/metric.py` — macro-F0.5 scorer
- `scripts/` — EDA, slicing, rule tuning, packaging, box runbook

### B. Additional Results

[Recall-vs-k table per view and country; OOF breakdown by number of true matches; feature
importance top-25; per-country test statistics; run times and peak memory.]
