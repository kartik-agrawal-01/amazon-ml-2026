# S2F runbook — beat Soha-2 (public LB 0.980795) on the A30, overnight 26→27 Sep

For a Claude Code session running ON the Jarvis A30 (repo at /home/amazon-ml-2026, `git pull` first). Written by HQ
26 Sep ~18:40 IST. Work phase by phase; after each phase append results to `runs/jarvis/S2F_NOTES.md` and push
(`bash scripts/jarvis/publish.sh` or plain git). Never upload anything yourself: humans upload, HQ advises.

## 0. What the grader rewards (official README + organisers' Q&A)
- Leaderboard = `matching_results.tsv` only: F0.5 per Source-1 entity, averaged over ALL S1. Singletons count: empty list
  = 1.0, any match = 0.0. A wrong match costs ~2x a missed one. Final ranking = private LB (rest of the test set);
  shortlisting looks at both leaderboards.
- `candidate_pairs.tsv` is not LB-scored, but "the approach that generates a smaller candidate set per Source 1 entity
  will be ranked higher in the final evaluation". Organisers' Q&A: in a cascade it is **the input to the FIRST scoring
  model**, and every matched ID must be in it.
- Allowed: pretrained open-weight models that are MIT/Apache-2.0, ≤ 8B parameters each, run offline and fine-tuned only
  on the provided data; unsupervised statistics on the unlabelled test files; **self-training and synthetic pairs made
  from the provided records**; small hand-written normalisation dictionaries. Forbidden: external data, APIs, geocoders,
  libpostal/gazetteers, hosted LLM APIs, hand-labelling.
- Top teams' zips are re-run: every change must be code in the repo that regenerates the file.

## 1. Where Soha-2 loses points
Validation F0.5 0.9865 (US/India) vs LB 0.981 ⇒ **France ≈ 0.95** (15% of test). In France she keeps 3.21 matches/S1
(US 3.36, India 3.34) and finds only 30% of the "different invented name at the same exact address, no other S1 with
that name or address" pairs (95% in US/India; 96–98% true on the full train). Her 0.80 threshold was tuned on US/India;
a country the model never saw is less confident. Test also has ~2x the unmatched S2/S3 records per S1 of train.

## Phase 0 — reproduce Soha-2 (gate, ≤ 2 h)
Inputs: her code under `soha/` (keep her files unchanged; changes go in new files `soha/s2f_*.py`) and her artifacts
copied to `/home/soha/` (models + cached tables).
1. Write `runs/jarvis/soha_map.md`: per stage (normalisation, TF-IDF blocking, bi-encoder retrieval, stage-1 scorer,
   cross-encoder, stage-2 LightGBM, stage-3 sibling model, threshold + one-to-one): script, inputs, outputs, cached?,
   time to recompute on this A30, which train S1 half trained the cross-encoder, which S1 form her validation split.
2. Reproduce her test output: ≥ 99.9% identical S1 sets vs `submissions/soha_matching_results_2.tsv` and her
   validation F0.5 (0.9865, per country).
3. Save the decision inputs (parquet, columns `s1, cand, country, p` with p = final stage-3 probability BEFORE the 0.80
   threshold and one-to-one; keep p2 = stage-2 probability too):
   `/home/s2f/val_pairs.parquet` (all candidate pairs of her validation S1), `/home/s2f/val_s1.txt` (ALL validation S1
   ids, one per line, including S1 with no candidates), `/home/s2f/test_pairs.parquet` (all test candidate pairs).
If a stage must be recomputed and it takes > 3 h, write that in LOG.md and continue from the cached parts only.

## Phase 1 — metric-aware decisions on her probabilities (≈ 1 h, CPU) → file D1
`src/hq_decide.py` (on main). One-to-one first, then per S1: top pair kept if p ≥ t_first, further pairs if
p ≥ t_rest. Tuned per country on validation; France (unseen) gets the tuned rule shifted down until its matches/S1
equal the US/India average on test (label-free, capped at 0.30).
```
python -m src.hq_decide tune  --val /home/s2f/val_pairs.parquet --val-s1 /home/s2f/val_s1.txt \
    --gt data/student_resource/dataset/train/train_ground_truth.tsv --out /home/s2f/rule.json
python -m src.hq_decide apply --test /home/s2f/test_pairs.parquet --rule /home/s2f/rule.json \
    --test-s1 data/student_resource/dataset/test/test_source1.tsv --out /home/s2f/d1_decide.tsv
python -m src.hq_keyfill --test-dir data/student_resource/dataset/test --matching /home/s2f/d1_decide.tsv \
    --out /home/s2f/d1_matching_results.tsv            # + --candidates/--out-candidates for the package
```
Report: validation F per country (base 0.80 vs tuned), t_first/t_rest per country, France delta and matches/S1,
France sure-key coverage, pairs added by the key fill. Validator → `submissions/s2f_d1_matching_results.tsv`.

## Phase 2 — exact-key rule features in her stage 2 (≈ 2–3 h) → file D2
`src/hq_keys.py`: keys for ALL records (same normalisation as `src/hq_keyfill.py: _keys_chunk`), `key_pairs` per
country with `kq_ctx` = ALL S1 of the country (train: all 2.2M S1, never a sample). `calibrate` P per (rule, country) on
the train S1 half NOT used to fit stage 2 (no leakage); France = min over US/India (`rules_for_country`). Features per
(s1, cand): key_rule code, key_p, ctx_core_other, ctx_addr_other, ncat code, acat code (0 when no key). Join onto her
stage-2 train/val/test tables, retrain stage 2 (her params/split), then stage 3. **Gate: validation F0.5 ≥ baseline −
0.0005 overall AND per country.** Then Phase-1 decisions (re-tune) + key fill → D2. Box evidence: these features gave
+0.48 pt on the unseen-country screen (US-trained → India).

## Phase 3 — France adaptation (allowed: self-training + synthetic pairs from provided records) (≈ 2–3 h) → D3
Her cross-encoder never saw French text. Continue fine-tuning it (1 epoch, same hyper-parameters, from her checkpoint)
on a mix: 50% her original training pairs + 50% French pairs built only from the test records:
- pseudo-positives: French test pairs with stage-3 p ≥ 0.97 and the one-to-one owner, plus French 'sure' key pairs
  (hq_keys rules with train P ≥ 0.96, record not 'sure' for 2+ S1);
- pseudo-negatives: French pairs with p ≤ 0.02 among each S1's top-10 candidates, and pairs whose record was given to
  another S1 with p ≥ 0.97;
- synthetic positives: French S1 records rewritten with the French noise seen in S2/S3 (docs/FRANCE_FIXES.md: "N° 32",
  zero-padded numbers, région ↔ département ↔ missing, St ↔ Saint, legal form moved/added/dropped (SARL, SAS, EI…),
  "& Fils"/"Groupe"/"Holding" appended, domain-name forms "xyzsas.com", accents dropped, word swaps, R./Av./Bd.).
Re-score FRENCH test pairs only with the adapted cross-encoder (same feature slot), rerun stages 2–3 for France,
decisions + key fill → D3. Checks: US/India validation F with the adapted cross-encoder must not drop (≥ −0.0005);
France matches/S1 → ≈ 3.35, empty rate ≈ 5.6–5.9%, sure-key coverage up.

## Phase 4 — a second, stronger cross-encoder as a stage-2 feature (only if Phases 1–3 are done; ≈ 3–4 h) → D4
multilingual-e5-base (MIT, 0.3B) fine-tuned as a pair classifier on the SAME cross-encoder half of the train S1 (no
leakage), scored on stage-2 train/val/test pairs, added next to her e5-small probability; retrain stages 2–3. KEEP only
if validation F0.5 +0.001. (bge-reranker-v2-m3, Apache-2.0, 0.6B, only if an H100 appears.)

## Phase 5 — smaller candidate set (final-ranking criterion, after the model is frozen)
Report mean/median candidates per S1 at the FIRST scoring model's input. If validation shows the stage-1 input can be cut
to the top-k by retrieval rank (k ≈ 15–25) with ≤ 0.05 pt validation F loss, produce the package from that setting.

## Outputs (every D-file)
Validator: `python data/student_resource/utils/validate_submission.py --matching … --candidate … --test-dir
data/student_resource/dataset/test` → `submissions/s2f_<tag>_matching_results.tsv` + `runs/jarvis/S2F_NOTES.md`
(validation F per country before/after, France stats, candidates/S1, commit) → push. HQ picks the uploads
(Day 3: 5 slots; keep the last slot for the best file).
Time plan: Phase 0 by ~21:30, D1 by ~22:30 (could still go in before midnight), D2 by ~02:00, D3 by ~05:00, D4 if
time. If a phase overruns by > 1 h, skip to the next and note it.
