# Experiment queue — top = next. Humans/HQ may edit (push from the laptop). The loop marks items DONE/FAILED.

HQ note (26 Sep 12:00): read docs/ENSEMBLE_AND_KEYS.md first. Two facts change how to evaluate. (1) The 8% slices hold
~12x fewer same-name decoys than the full data. Blocking, cascade, one-to-one and key-rule changes can look neutral on
the slice Q and still matter at full density. For those items the metric is the FULL-DENSITY train pass (150K train S1
against all docs of the country, OOF F0.5 per country + pair recall after the cascade), plus test statistics per
country. (2) Train GT has 3.46 true matches per S1 in both US and India. v2 predicts 3.35 (US), **3.12 (India)**,
3.30 (France).

UPLOAD POLICY (HQ, 26 Sep 12:30): uploads are spent only on files expected to beat the best LB by ≥ 1 pt. So do
NOT run a full-data test pass per item: implement and evaluate items 2, 3, 4 (and 5 if ready) separately on the
full-density train pass / slices, then run ONE full-data test pass with every KEEP → v3 = France fixes + global
one-to-one (items 2–3). Key candidates, reverse blocking and name_ph at FULL density run on the Jarvis lane
(scripts/jarvis/QUEUE.md). The box's v3 is the fallback if Jarvis fails.
ens2c (LOG #3: one-to-one + key pairs on top of v2) is HELD, not uploaded.

1. [P0] FAST LANE + correctness gate (see DAY_TASK.md). Nothing else until it passes.
2. [P0.5, correctness — KEEP without the Q rule] GLOBAL one-to-one. v2 applies one-to-one per 100K-S1 block only. In
   the v2 submission, 13,806 S2/S3 ids are assigned to 2–5 S1s (28,302 pair slots, ≥ 14.5K guaranteed false positives;
   France 48%). The OOF never showed it because the train pass is one block per country. Fix: in the test pass, buffer
   each country's decided pairs WITH p across all blocks, then `src/hq_keys.py: global_one_to_one(q_rid, c_rid, p)`
   (keep the max-p S1 per record) before writing matching_results.tsv (candidate_pairs.tsv unchanged). Verify on the
   fast lane (v2 rescored → 0 multi-owner ids; report pairs removed per country) and use it in every later file.
2b. [Evidence, cheap, ungated, ~10 min of light work; do it while a heavy job runs] Why does v2 miss the 'sure' key
   pairs? HQ exported them (computed with the NEW normaliser over all test records):
   `scripts/hq_ens/sure_pairs_missed_by_v2.tsv.gz` (130,678 pairs: source1_entity_id, entity_id, country, rule, flags).
   For each pair, look up output_v2/candidate_pairs.tsv and tag it: in v2's candidate set (the model rejected it) or
   not (blocking/cascade cut it). Write counts by rule and country to `runs/day/keys_check.md` (run it as
   `python scripts/<name>.py` and stream candidate_pairs.tsv, keeping only the listed S1 ids: guard rules in
   DAY_TASK.md). HQ decides from this
   which rules to force: cut pairs are ~96% true at the train rates, model-rejected ones may be the real negatives.
   The key rows: India `core_eq|num_eq` (29.8K pairs, train rate 0.958) and France `disjoint|a_eq|invented` (24.8K).
   — DONE 26 Sep 12:20 (loop): runs/day/keys_check.md. Cut before the model (blocking/cascade): France 71%, India 48%, US 16%.
2c. [Evidence, cheap, ungated, ~5 min of light work while a heavy job runs — HQ needs it for tonight's upload call]
   v2's OOF PER COUNTRY. v2's log only has the overall 0.9623 (150K train S1: US 89,969 / India 60,031). If
   `output_v2_train/oof_pairs.tsv.gz` (or any dump of v2's train-pass OOF pairs) exists, compute OOF F0.5 per country
   at thr 0.70 + one-to-one on that sample, plus predicted matches/S1 and empty rate per country →
   `runs/day/v2_oof_by_country.md`. If no dump exists, write that in one line and stop (do NOT retrain v2).
   — DONE 26 Sep 15:45 (loop): runs/day/v2_oof_by_country.md. OOF India 0.9476 / US 0.9722; India cand recall 0.919 vs US 0.976.
2d. [France-relevant, light work on saved probs — HQ needs it by ~21:30 for the jv1 France threshold] Unseen-country
   threshold (follows your 15:13 finding and NEXT 2). On the cross-country screens with --save-probs outputs (xc_in
   from n3xin, xc_us from the next xc_us run), for the TARGET country:
   (i) hidden F0.5 vs threshold 0.30–0.95, step 0.05 (thr_sweep.py);
   (ii) three LABEL-FREE rules for picking the target's threshold, each scored by the hidden F0.5 it gives:
        R1 count matching: t where the target's predicted matches/S1 = (source OOF predicted matches/S1 ÷ source GT
           matches/S1, both at the OOF threshold) × the target's GT matches/S1 (for France we assume 3.46 as in US/India);
        R2 empty matching: t where the target's predicted-empty rate = the source's OOF predicted-empty rate;
        R3 sure-key matching: t where the target's recall of its 'sure' exact-key pairs (src/hq_keys key_pairs +
           apply_rules, p_min 0.96, on the slice) = the source's at its OOF threshold.
   Table per screen: OOF-chosen t, hidden-best t, R1/R2/R3 t, hidden F at each → `runs/day/unseen_thr.md`. If a rule
   lands within 0.001 of the hidden best on both screens, HQ applies it to France in jv1 (Jarvis QUEUE 1b).
   — DONE (i)+(ii) 26 Sep 16:50 (loop): runs/day/unseen_thr.md. No plain rule within 0.001 on both screens; R2 'lower-only'
     = min(t_oof, R2) is: xc_us +0.0158, xc_in 0, mix +0.0005. It leaves v2's France threshold unchanged. R3 not done yet.
   — DONE (iii) R3 26 Sep 19:20 (loop): unseen_thr.md §(iii). Two-sided R3 unstable (recall saturates -> t 0.02/0.99);
     lower-only R3n = R2 lower-only (xc_us +0.0158, xc_in 0, mix +0.0002). Nothing beyond --thr-adapt.
3. [France, HQ-DONE diagnosis → APPLY + EVALUATE] HQ's France fixes are in `src/normalize.py` + `src/features.py`
   (commit 7defba7 "france: normalisation fixes"; write-up `docs/FRANCE_FIXES.md`). Bugs fixed: "N° 32" normalised
   to "north 32" (5% of French addresses); zero-padded house numbers (also 3% of US / 5% of India S2/S3); région vs
   département never agreeing; St-Nazaire → "street nazaire"; legal form lost when it leads the name; EI not a legal
   form; French filler words (groupe/fils/associés/développement/…) counted as content words in extra_*_content. On
   French test pseudo-pairs, same-name pairs with identical normalised address go from 14% to 33%. Steps:
   a) NORMALISATION CHANGED ⇒ rebuild the per-country stores (new cache dirs; never delete v2's) for the slices and
      the full data. Candidate sets built before stay valid.
   b) Slice Q (expect ≥ Q0; the leading-zero fix may help US/India a little) AND the full-density train pass (OOF per
      country must not drop).
   c) Full data via the fast lane + global one-to-one (item 2) → validator → `submissions/v3_matching_results.tsv` →
      `runs/v3/NOTES.md`. Include a France table: empty rate (v2 5.2%), mean matches (v2 3.30), and **coverage of the
      French 'sure' key pairs by rule** (`src/hq_keys.py`: key_pairs + apply_rules, p_min 0.96; v2 covers only 34% of
      `disjoint|a_eq|invented` in France vs 92% in US; target ≥ 85%). SUBMIT-READY if the train-pass OOF holds; HQ
      decides the upload.
   d) Save per-pair test probabilities (P0-c).
4. [Keys + reverse blocking — IMPLEMENTED on main by the box (5b8720f, 628591f, f4fd6f1): the box OWNS this code.]
   The box only slice-screens it (chain3: n3k, n3ph, n3r). The FULL-DENSITY runs and the full-data files belong
   to the Jarvis lane (scripts/jarvis/QUEUE.md item 1, same flags), so don't start a full-data run of them here.
   Watch `runs/jarvis/ISSUES.md` on main: Jarvis files the minimal fixes it needed at full density there (it
   commits them on its branch). Port each fix to main in your next session and note it in runs/day/LOG.md.
5. [India recall — moved up; the largest measured loss] India: 3.12 predicted matches/S1 vs 3.465 in the train GT (US
   3.35 vs 3.459). That is ≈ 0.3 true pairs/S1 missing on 47% of test. Exact keys don't fix it: India's key-rule
   coverage is already 93–99%, so the loss is fuzzy (Indic scripts, landmark addresses, heavy reordering). Full-density
   blocking recall is 0.933 at k=10 (US 0.984), and the cascade loses more.
   a) Decompose India's missed GT pairs on the full-density train pass: blocking miss / cascade cut / model FN / model
      FP, by category (Indic-script name, empty candidate address, name-token Jaccard bins, address Jaccard bins,
      transliterated legal forms). Write `runs/day/india_recall.md`.
      — DONE 26 Sep 15:50 (loop): runs/day/india_recall.md. Lost 11.4% of GT = blocked 6.7% / cascade 1.4% / model FN 3.3%;
        Indic cand names 20.5% blocked; empty cand address TP rate 43%.
   b) Slice screens on the box (n3ph = name_ph back, n3r = reverse blocking) are fine. The full-density
      evaluation runs on Jarvis (item 1: 6 views, k 15, keys + reverse, 600K train S1).
6. [France] Country-neutral model: adversarial validation (classifier France-vs-US/India pairs on the pair features),
   drop or re-normalise the most country-shifted features (e.g. rank/percentile within country instead of raw
   counts); evaluate Q. Also: 36% of French S2/S3 addresses have no region/département while S1 always has one
   (US 4.5%, India 14%) — consider imputing the region code from the city using S1 itself (no external data).
   — PARTIAL 26 Sep 18:55 (loop): adversarial validation only -> runs/day/adv_val.md. AUC 0.997 France vs rest, shift spread
     over address-format + decoy-density features (still 0.985+ after dropping the top 5) -> feature dropping is low-EV;
     the density shift is what --thr-adapt handles. Region imputation part not done.
   — DONE (evidence) region imputation 26 Sep 19:40 (loop): runs/day/fr_region_impute.md. 74% of region-less FR S2/S3
     imputable from 22 S1 cities at 100% label-free accuracy; not implemented (unscorable on Q, v3 queued) — HQ's call.
7. [Accuracy, cheap] Model capacity on the fast lane: more trees/leaves, train S1 150K → 300K, 3-seed average.
8. [Efficiency] cascade_top 10 → 8 only if the full-density pair recall after the cascade holds PER COUNTRY. Every
   French S1 already sits at the cap, so consider a per-country cap before any global cut.
9. [Speed] Profile the full-data run by stage; vectorise the slowest stage-B features.
