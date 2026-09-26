# Experiment queue — top = next. Humans/HQ may edit (push from the laptop). The loop marks items DONE/FAILED.

1. [P0] FAST LANE + correctness gate (see DAY_TASK.md). Nothing else until it passes.
2. [France, HQ-DONE diagnosis → APPLY + EVALUATE] HQ diagnosed France label-free on the raw test files and pushed the
   fixes in `src/normalize.py` + `src/features.py` (commit "france: normalisation fixes"; full write-up in
   `docs/FRANCE_FIXES.md` — read it first). Headline bugs found: "N° 32" was normalised to "north 32" (5% of French
   addresses), zero-padded house numbers (also 3% of US / 5% of India S2/S3), region vs département never agreeing,
   St-Nazaire → "street nazaire", legal form lost when it leads the name, EI not a legal form, French filler words
   (groupe/fils/associés/développement/…) counted as content words in extra_*_content.
   Pseudo-pair proxies on French test data: same-name pairs with identical normalised address 14% → 33%, address
   token Jaccard 0.655 → 0.818. Steps:
   a) `git pull`; NORMALISATION CHANGED ⇒ rebuild the per-country stores for data_slice / data_xc_us / data_xc_in
      (delete `cache_*/…parquet` + `.done.json` or use a new cache dir) — candidate sets built before stay valid.
   b) Evaluate Q (expect ≥ Q0; the leading-zero fix should help US/India a little). KEEP/REVERT per the rule.
   c) If KEEP: full-data via the fast lane (rebuild the full store, rescore v2's candidate set with recomputed
      features + retrained model) → validator → `submissions/v3_matching_results.tsv` → `runs/v3/NOTES.md` with the
      France table: predicted-empty rate (v2 5.2%), mean matches (v2 3.30), mean max-prob per S1, and the share of
      predicted pairs whose n_addr agree on the region code. SUBMIT-READY if Q ≥ Q0 − 0.0005 (France cannot be
      measured; the decision is HQ's).
   d) Save per-pair test probabilities (P0-c) so HQ can compare France score distributions v2 vs v3.
3. [France] Country-neutral model: adversarial validation (classifier France-vs-US/India pairs on the pair features),
   drop or re-normalise the most country-shifted features (e.g. rank/percentile within country instead of raw
   counts); evaluate Q. Also: 36% of French S2/S3 addresses have no region/département while S1 always has one
   (US 4.5%, India 14%) — consider imputing the region code from the city using S1 itself (no external data).
4. [Accuracy, cheap] Model capacity on the fast lane: more trees/leaves, train S1 150K → 300K, 3-seed average.
5. [India recall] Slice: India-only extra view (name_ph / transliterated / dense MiniLM via src/blocking_b.py —
   see runs/night/track_b_block_eval.txt) or India k 10→15, keeping final cands/S1 ≤ 10 after the cascade;
   measure India pair recall + Q. If it wins → candidate-augmentation full-data run (India only).
6. [Efficiency] cascade_top 10 → 8 or 6 if Q holds (smaller candidate sets rank higher).
7. [Speed] Profile the full-data run by stage; vectorise the slowest stage-B features.
